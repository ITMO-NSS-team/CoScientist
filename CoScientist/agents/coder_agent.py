from openhands.sdk import conversation
import os
from pathlib import Path
import platform
import time
import subprocess
from typing import Generator, Any
import traceback
import httpx
from pydantic import BaseModel, Field, SecretStr

from google.adk import Context, Workflow, Event
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
import signal
from google.adk.workflow import node
from google.adk.events import RequestInput

from openhands.sdk import LLM, Conversation, get_logger
from openhands.workspace.docker.workspace import find_available_tcp_port
from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation
from openhands.sdk.utils.command import execute_command
from openhands.tools.preset.default import get_default_agent
from openhands.workspace import DockerWorkspace

logger = get_logger(__name__)

CONTAINER_NAME = "coscientist-coder-sandbox"
_active_workspace: "PersistentDockerWorkspace | None" = None


class SandboxResult(BaseModel):
    execution_summary: str = Field(description="A brief report on the work done by the agent.")

def detect_platform():
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


class PersistentDockerWorkspace(DockerWorkspace):
    """DockerWorkspace subclass that does NOT auto-remove the container."""

    _reused: bool = False

    def _get_container_state(self) -> str | None:
        """Return the container state ('running', 'exited', etc.) or None."""
        result = execute_command(
            ["docker", "inspect", "-f", "{{.State.Status}}", CONTAINER_NAME],
            print_output=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None

    def _try_reuse_existing(self, context: Any) -> bool:
        """Attach to an existing container and restart it to reset the server."""
        state = self._get_container_state()
        if state is None:
            return False

        logger.info(f"Restarting existing container '{CONTAINER_NAME}' (state: {state}) to reset the agent server...")
        if state == "paused":
            execute_command(["docker", "unpause", CONTAINER_NAME], print_output=False)
        
        result = execute_command(["docker", "restart", CONTAINER_NAME])
        if result.returncode != 0:
            logger.warning(f"Failed to restart container: {result.stderr}")
            return False

        id_result = execute_command(
            ["docker", "inspect", "-f", "{{.Id}}", CONTAINER_NAME],
            print_output=False,
        )
        if id_result.returncode != 0:
            return False

        object.__setattr__(self, "_container_id", id_result.stdout.strip())
        object.__setattr__(self, "_image_name", self.server_image)

        port_result = execute_command(
            ["docker", "inspect", "-f",
             "{{(index (index .NetworkSettings.Ports \"8000/tcp\") 0).HostPort}}",
             CONTAINER_NAME],
            print_output=False,
        )
        if port_result.returncode == 0 and port_result.stdout.strip():
            object.__setattr__(self, "host_port", int(port_result.stdout.strip()))

        object.__setattr__(self, "host", f"http://127.0.0.1:{self.host_port}")
        object.__setattr__(self, "api_key", None)

        if self.detach_logs:
            import threading
            _logs_thread = threading.Thread(
                target=self._stream_docker_logs, daemon=True
            )
            object.__setattr__(self, "_logs_thread", _logs_thread)
            _logs_thread.start()

        self._wait_for_health(timeout=self.health_check_timeout)
        logger.info(f"Docker workspace reconnected at {self.host}")

        super(DockerWorkspace, self).model_post_init(context)
        self._reused = True
        return True

    def _start_container(self, image: str, context: Any) -> None:
        if self._try_reuse_existing(context):
            return

        execute_command(["docker", "rm", "-f", CONTAINER_NAME], print_output=False)

        self._image_name = image

        if self.host_port is None:
            from openhands.workspace.docker.workspace import find_available_tcp_port
            self.host_port = find_available_tcp_port()
        else:
            self.host_port = int(self.host_port)

        from openhands.workspace.docker.workspace import check_port_available
        if not check_port_available(self.host_port):
            raise RuntimeError(f"Port {self.host_port} is not available")

        if self.extra_ports:
            if not check_port_available(self.host_port + 1):
                raise RuntimeError(f"Port {self.host_port + 1} is not available for VSCode")
            if not check_port_available(self.host_port + 2):
                raise RuntimeError(f"Port {self.host_port + 2} is not available for VNC")

        docker_ver = execute_command(["docker", "version"]).returncode
        if docker_ver != 0:
            raise RuntimeError("Docker is not available.")

        flags: list[str] = []
        for key in self.forward_env:
            if key in os.environ:
                flags += ["-e", f"{key}={os.environ[key]}"]

        for volume in self.volumes:
            flags += ["-v", volume]
            logger.info(f"Adding volume mount: {volume}")

        ports = ["-p", f"127.0.0.1:{self.host_port}:8000"]
        if self.extra_ports:
            ports += [
                "-p", f"127.0.0.1:{self.host_port + 1}:8001",
                "-p", f"127.0.0.1:{self.host_port + 2}:8002",
            ]
        flags += ports

        if self.enable_gpu:
            flags += ["--gpus", "all"]

        if self.network:
            flags += ["--network", self.network]

        flags += ["--shm-size", "2g"]

        run_cmd = [
            "docker", "run", "-d",
            "--platform", self.platform,
            "--ulimit", "nofile=65536:65536",
            "--name", CONTAINER_NAME,
            *flags,
            image,
            "--host", "0.0.0.0",
            "--port", "8000",
        ]
        proc = execute_command(run_cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to run docker container: {proc.stderr}")

        self._container_id = proc.stdout.strip()
        logger.info(f"Started NEW container: {self._container_id} (name={CONTAINER_NAME})")

        if self.detach_logs:
            import threading
            self._logs_thread = threading.Thread(
                target=self._stream_docker_logs, daemon=True
            )
            self._logs_thread.start()

        if not self.host:
            object.__setattr__(self, "host", f"http://127.0.0.1:{self.host_port}")
        object.__setattr__(self, "api_key", None)

        self._wait_for_health(timeout=self.health_check_timeout)
        logger.info(f"Docker workspace is ready at {self.host}")

        super(DockerWorkspace, self).model_post_init(context)

    def cleanup(self) -> None:
        self._stop_logs.set()
        if self._logs_thread and self._logs_thread.is_alive():
            self._logs_thread.join(timeout=2)
        logger.info(f"Container '{CONTAINER_NAME}' is kept alive for reuse.")

    def force_cleanup(self) -> None:
        if self._container_id:
            self._stop_logs.set()
            if self._logs_thread and self._logs_thread.is_alive():
                self._logs_thread.join(timeout=2)

            logger.info(f"Stopping and removing container: {CONTAINER_NAME}")
            execute_command(["docker", "stop", CONTAINER_NAME])
            execute_command(["docker", "rm", "-f", CONTAINER_NAME])
            self._container_id = None

        if self.cleanup_image and self._image_name:
            logger.info(f"Deleting Docker image: {self._image_name}")
            execute_command(["docker", "rmi", "-f", self._image_name])
            self._image_name = None


def _prompt_container_action():
    import sys
    if os.environ.get("COSCIENTIST_WEB_MODE") == "true" or not sys.stdin.isatty():
        logger.info("Web or non-interactive environment detected. Automatically cleaning up sandbox container.")
        return "cleanup"

    print("\n" + "=" * 60)
    print("Sandbox execution finished!")
    print("=" * 60)
    print("  Choose what to do with the container:")
    print("    [1] Remove container (clean slate next time).")
    print("    [2] Keep container running (preserve installed deps). Don't forget to close it later")
    print("=" * 60)
    while True:
        try:
            choice = input("  Your choice [1/2] (default: 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            return "cleanup"
        if choice in ("", "1"):
            return "cleanup"
        elif choice == "2":
            return "keep"
        else:
            print("  Please enter 1 or 2.")


def _extract_summary(conversation, received_events: list) -> str:
    from openhands.sdk.event.llm_convertible.action import ActionEvent

    try:
        agent_summary = conversation.ask_agent(
            "Summarize everything you did during this session in detail: "
            "what code was written, what dependencies were installed, "
            "what commands were executed, what results were obtained, "
            "and what files were created or modified. "
            "Be as detailed as possible."
        )
        if agent_summary and agent_summary.strip():
            logger.info("ABC Got agent-generated summary via ask_agent()")
            return agent_summary.strip()
    except Exception as e:
        logger.warning(f"ask_agent() failed, falling back to event-based summary: {e}")

    action_summaries = []
    for event in received_events:
        if isinstance(event, ActionEvent) and getattr(event, "summary", None):
            action_summaries.append(f"• {event.summary}")

    if action_summaries:
        steps = "\n".join(action_summaries[-20:])
        return f"Agent completed {len(action_summaries)} actions. Last steps:\n{steps}"

    return (
        f"Program completed. Last 5 events: "
        f"{', '.join(type(e).__name__ for e in received_events[-5:])}."
    )


async def run_openhands_sandbox(code: str = "") -> Any:
    """Execute code in a persistent Docker sandbox.

    Args:
        code: The task description or code to execute in the sandbox.
    """
    import asyncio

    def _run_sync():
        global _active_workspace

        api_key = os.getenv("LLM__CODER_KEY")
        llm_model = os.getenv("LLM__CODER_MODEL")
        base_url = os.getenv("LLM__MAIN_URL")
        server_image = os.getenv("DockerWorkspace__server_image") or "ghcr.io/openhands/agent-server:latest-python"

        llm = LLM(
            usage_id="agent",
            model=llm_model,
            base_url=base_url,
            api_key=SecretStr(api_key),
        )

        logger.info(f"Starting DockerWorkspace with image: {server_image}")
        
        desired_port = 8010
        try:
            from openhands.workspace.docker.workspace import check_port_available
            if not check_port_available(desired_port):
                raise RuntimeError
            host_port = desired_port
        except Exception:
            host_port = find_available_tcp_port()
            logger.info(f"Selected alternative host port {host_port} for coder sandbox")

        workspace = PersistentDockerWorkspace(
            server_image=server_image,
            host_port=host_port,
            platform=detect_platform(),
            extra_ports=True,
            enable_gpu=False,
            volumes=[f"{Path.cwd().parent}/workspace:/workspace:rw"],
        )
        _active_workspace = workspace
        
        try:
            agent = get_default_agent(llm=llm, cli_mode=True)
            
            received_events: list = []
            def event_callback(event) -> None:
                received_events.append(event)
                logger.info(f"Callback event: {type(event).__name__}")

            conversation = Conversation(
                agent=agent,
                workspace=workspace,
                callbacks=[event_callback],
            )
            assert isinstance(conversation, RemoteConversation)

            vscode_port = (workspace.host_port or 8010) + 1
            try:
                response = httpx.get(
                    f"{workspace.host}/api/vscode/url",
                    params={"workspace_dir": workspace.working_dir},
                )
                vscode_data = response.json()
                vscode_url = vscode_data.get("url", "").replace(
                    "localhost:8001", f"localhost:{vscode_port}"
                )
            except Exception:
                folder = f"/{workspace.working_dir}".replace("//", "/")
                vscode_url = f"http://localhost:{vscode_port}/?folder={folder}"

            logger.info(f"VSCode URL: {vscode_url}")

            task_code = code
            logger.info(f"Task to send to sandbox agent (len={len(task_code)}): {task_code[:200]!r}")
            conversation.send_message( # Агент возможно передеет на А2А, пусть промпт и другие параметры пока будут здесь
                f"### System Guide for Task Execution\n\n"
                f"#### 1. Task Description\n"
                f"Execute the following task:\n"
                f"{task_code}\n\n"
                f"#### 2. Workflow and Execution Order\n"
                f"- Step 1: Write all the necessary code first.\n"
                f"- Step 2: Create a pyproject.toml file before installing any dependencies.\n\n"
                f"#### 3. Technical Requirements\n"
                f"- Package Manager: Use uv exclusively for all dependency installations.\n"
                f"- Environment Isolation: All packages must be installed strictly within a virtual environment (venv).\n"
                f"- Computing Resources: Use a GPU if available.\n\n"
                f"#### 4. Artifact Structuring\n"
                f"All outputs must be saved into strictly defined directories:\n"
                f"- 'code' - for source code scripts.\n"
                f"- 'results' - for plots, visualizations, and final results.\n"
                f"- 'artefacts' - for trained models and intermediate artifacts."
            )
            conversation.run(timeout=10800.0)

            summary = _extract_summary(conversation, received_events)
            conversation.close()

        except KeyboardInterrupt:
            logger.warning("Execution interrupted by user (Ctrl+C). Cleaning up container.")
            summary = "Execution interrupted by user."
            if _active_workspace:
                _active_workspace.force_cleanup()
                _active_workspace = None
        except Exception as e:
            logger.error(f"Error during sandbox execution: {e}")
            tb = traceback.format_exc()
            summary = f"Execution failed: {e}\n{tb}"

        action = _prompt_container_action()

        if action == "cleanup":
            workspace.force_cleanup()
            _active_workspace = None
            logger.info("Container removed.")
        else:
            logger.info(f"Container '{CONTAINER_NAME}' kept alive. "
                        "It will be reused on the next run.")

        return SandboxResult(execution_summary=summary)

    try:
        return await asyncio.to_thread(_run_sync)
    except asyncio.CancelledError:
        logger.warning("Execution cancelled. Initiating cleanup of active workspace...")
        if _active_workspace:
            try:
                # We use asyncio.shield to prevent the cleanup itself from being cancelled
                # if the server is shutting down aggressively.
                await asyncio.shield(asyncio.to_thread(_active_workspace.force_cleanup))
            except Exception as e:
                logger.error(f"Error during cleanup on cancel: {e}")
        raise

async def stop_coder_container() -> str:
    import asyncio
    global _active_workspace
    if _active_workspace:
        try:
            await asyncio.to_thread(_active_workspace.force_cleanup)
            _active_workspace = None
            msg = f"Container '{CONTAINER_NAME}' has been forcefully stopped and removed."
            logger.info(msg)
            return msg
        except Exception as e:
            err = f"Failed to stop container: {e}"
            logger.error(err)
            return err
    else:
        return "No active coder sandbox container to stop."

run_openhands_sandbox = FunctionTool(run_openhands_sandbox)
stop_coder_container = FunctionTool(stop_coder_container)