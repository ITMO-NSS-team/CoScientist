"""Unified command-line entry point for CoScientist.

One dispatcher for every way to run the system — prefer this over the
individual module entry points:

    python -m CoScientist web                      # web UI (uvicorn)
    python -m CoScientist cli                       # interactive REPL
    python -m CoScientist a2a all                   # all A2A agent servers
    python -m CoScientist a2a serve <key>           # one agent over A2A
    python -m CoScientist a2a bench --agent .. --text ..

Every heavy import is done lazily inside its handler. Some run modes (the A2A
servers) apply per-agent environment defaults BEFORE their agent/tool modules
load, so nothing agent-related may be imported at module top here. Keeping the
top of this module import-light also makes ``--help`` fast.

The legacy entry modules (``CoScientist.main``, ``CoScientist.web.server``,
``CoScientist.a2a.*``) still work and now forward here / are delegated to from
here, so existing commands and muscle memory keep working.
"""
import argparse

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------
def run_web(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Serve the FastAPI web interface via uvicorn."""
    import uvicorn

    # Pass the import string (not the app object) so ``--reload`` can re-import
    # the worker. The app itself lives in CoScientist.web.server:app.
    uvicorn.run(
        "CoScientist.web.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def run_repl() -> None:
    """Interactive terminal REPL against the in-process agent system."""
    import asyncio

    from CoScientist.main import create_manager

    async def _loop() -> None:
        manager = await create_manager()
        print("CoScientist (ADK) initialized — type 'exit' to quit.")
        print("(The web UI is also available: python -m CoScientist web)\n")
        try:
            while True:
                query = input("Enter query (or 'exit'): ")
                if query.lower() in {"exit", "quit"}:
                    break
                result = await manager.run(query)
                print("\n=== Final Response ===")
                print(result)
                print()
        finally:
            await manager.close()

    asyncio.run(_loop())


def _run_a2a(a2a_cmd: str, rest: list) -> None:
    """Delegate to the A2A modules, which own the env-before-import ordering.

    ``rest`` is forwarded verbatim to the underlying module's argparse, so e.g.
    ``a2a serve research --host 0.0.0.0`` and
    ``a2a bench --agent research --text "hi"`` keep their original flags.
    """
    if a2a_cmd == "all":
        import asyncio

        from CoScientist.a2a import run_all

        asyncio.run(run_all.main())
    elif a2a_cmd == "serve":
        from CoScientist.a2a import serve

        serve.main(rest)
    elif a2a_cmd == "bench":
        import asyncio

        from CoScientist.a2a import benchmark

        asyncio.run(benchmark.main(rest))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m CoScientist",
        description="CoScientist unified entry point.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    web = sub.add_parser("web", help="Run the web UI (uvicorn).")
    web.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    web.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    web.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")

    sub.add_parser("cli", help="Interactive terminal REPL.")

    # The a2a sub-command takes a mode plus arbitrary trailing flags that are
    # forwarded to the underlying module (collected via parse_known_args below,
    # so flags like --agent/--text/--host survive regardless of position —
    # argparse.REMAINDER would drop a leading "--flag").
    a2a = sub.add_parser(
        "a2a",
        help="Run A2A agent servers / benchmark client.",
        description=(
            "a2a all            serve all A2A agents in one process\n"
            "a2a serve <key>    serve one agent over A2A [--host HOST]\n"
            "a2a bench ...      smoke-test/perf client (--agent <key> --text <msg> ...)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    a2a.add_argument("a2a_cmd", choices=["all", "serve", "bench"], help="A2A run mode.")
    return parser


def main(argv=None) -> int:
    load_dotenv()
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)

    if args.cmd in ("web", "cli") and rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")

    if args.cmd == "web":
        run_web(args.host, args.port, args.reload)
    elif args.cmd == "cli":
        run_repl()
    elif args.cmd == "a2a":
        _run_a2a(args.a2a_cmd, rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
