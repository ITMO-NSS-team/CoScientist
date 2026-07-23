Now that we've defined the broad direction, here are more practical tasks I'd like to focus on for the paper research progress

0. For the llm calls, we are going to need proxy. here's the example for simple openai
from openai import OpenAI
from httpx import Client

client = OpenAI(
    api_key= OPENAI_KEY,
    http_client=Client(proxy=HTTP_PROXY)
)
We are going to need such an option within coscientist and alembic. Also, right now the LLM setup in coScientist and alembic module are split apart, we need a way to set them up in a unified way. Also make sure there are safety mechanisms in place for everython not to break if api suddenly does not respond. I know we have that in alembic, coscientist not so sure.  

1. We want to launch this cluster part of coscientist for experiment: CoScientist gets a task along with needed repo(s), and provided data (if applicable), returns result.

2. This i am not acutally sure about, the workflow is more of a rough idea now: orchestrator/planner develop the plan, then the flow is passed on to some kind of experiment control that checks whether the alembic has already converted the required module and the executor agent actually calls the required mcp's tools, may code intermediate steps if needed and submits the results. Coder's functionality must also include tool calling for that.

3. We want to use only the modules we need, so similar to CoScientist/agents/microfluidics.yaml and CoScientist/agents/system.yaml we need a config for our experimental setup

4. We want to adapt our system to the available benchmarks. TMBench was very sloppy in terms of tasks (1000-iteration CPU training in task nnunet_train_model) and the inner testing phase structure (alembic required a lot of adaptation to fit into the changed environment, run script that called http server etc) however had good gold definition - this might pose a potential compatability gap with CORE-Bench's capsules with the dockerfile, in all such cases we just compare our results to the heldout gold, taking as much of the evaluation code a spossible from the bench. Basically we want to supply our system with all the data and verify against well-defined gold, not struggle to adapt our soultion to setup within the bench if that is the case. If not - of course use what is already provided.

5. When the setup of CoScientist is ready for the experiments - we want to first test on a few tasks to check the system works and is stable before moving on to further system 

6. Our benchmark run setup must check whether the system is capable of GPU usage, using all of the resources if available. We are expecting our servers to be either very ram-capable or have a good GPU. that's why we'd also ideally want to split our tasks between the ones that are definitely going to require GPU, and the ones that are perfectly fiine with cpu only - and do those in parallel.

7. The benchmarks we are looking forward to: TMBench -> ToolArena (big overlap with TMBench) -> ScienceAgentBench -> CORE-Bench. These are in the order of increasing amount of tasks - as we have limited time to do those. Perform all research requirted for those to be ran with our system in advance.

8. We want to baseline against an open coder agent - openhands + our main model that just performs the task straight away in an isolated environment (within a docker container)

9. We want all of the required alembic mcp converions to be performed in advance and for them to be available in CoScientist's MCP/tool registry - Retrieve_tools tool that searches from base, right now it is available to orchestrator and retriever -so that the task executions themselves are more streamlined. So, for a bench we define a set of tools we need to convert beforehand + a rough sketch of what they are going to be used for, for the alembic to have tools that would most closely 

10. Early stage task: test that coscientist actually sees and is able to call the servers that alembic serves - and the tool call resaults are retrieved accurately, check on a simple task from tmbench

11. Coscientist should prioritize alembic available MCP's over plain coder, that is a fallback - using the execution graph (it should be available in coscientist) we want to track that

12. We want to track token usage, amount of steps etc. All that is already available for alembic's bench runs, ensure parity with CoScientist

13. Our main model - glm 5.2