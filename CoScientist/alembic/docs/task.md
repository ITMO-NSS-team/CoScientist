/home/stas/Documents/GitHub/CoScientist/CoScientist/alembic/docs/paper/emnlp2026_demo.pdf - Here's demo paper that we've submitted for our project Alembic CoScientist/alembic. These are pretty much all the results and finding we have with it now, and we want to extend it to a full main track AAAI27 paper. With this research we want to get the answers to some of our questions and define the development vector before starting the development.

Don't answer all at once. This includes research, assessment, planning and asking me questions when necessary.


What we compared to in the demo:
C1. [ToolMaker's](/home/stas/Documents/GitHub/ToolMaker/2502.11705v2.pdf) bench is concise and robust with tests and invocations but is actually terrible to run and constantly breaks, as well as requiring adaptation to certain structure to test just an individual function script (single tool to complete target task was their initial architecture) - far away from a benchmark for MCP's specifically.

C2. [ToolRosella's](/home/stas/Documents/GitHub/ToolRosella-main/2603.09290v5.pdf) 122-repo set is interesting but
has no option to assess MCP CONVERSION quality, as those do not ship with expert-made gold set mcp's or premade tests - basically they test MCP conversion with "three endpoints revealed" criterion, we want to do better.  Their solution consisted of processing user requests, converting repos from pool to mcp, and searching and using tools to solve concrete tasks. 


Ideas that are in my head now:

I1. We want focus on SPECIFIC scienctific repos. we want to be able to convert any repo, but the initial concept is making the often poorly documented or hard to setup and actually valuable (maybe in in relation to some concrete paper). It is not intersting to convert RDKit for example because it is already represented in pre-training data of most large LLM's and the conversion is kind of useless there. ToolRosella's bench is not fully relevant in this way, ToolMaker has an interesting repo selections but again terrible to launch and goes not focus on MCP.

I2. Because LLM's have probably seen all of the GitHub in their pre-train phase, we need to somehow proof that our soultion is actually needed for something. The lesser know scientific repos are probably still present there, but at least not as much as RDKit.

I3. One argument could be made is that Run-to-run consistency of calling a coder agent on a niche scientific repo may not be consistent - it may solve the same wording of a problem in slightly different ways, followed by stochastically varying exploration paths may yield different results - while MCP approach allows to reuse the same code between runs and between points of access (people, autonomous AI systems etc.), and token economy.

I4. ToolRosella task part is kinda good, as the converted MCP's get actually used to complete the task, we may want to extend our experiment scope to actually use our MCP's to complete some sort of task set, while still not compromising in assessing the quality and robustness of MCP itself.

I5. We see our direct competition not much the existing repos, but the open sandbox coding agents (coder agent v mcp approach) - does our soultion really have a meaning or arguments I2-I3 don't stand and frontier model on a good harness completes any task on niche repoitories.

I want you to research:

Field state

F1. Relevant fresh soultion related to code -> mcp, better peer reviewed (toolrosella still isn't and
their success criteria shows)

F2. Relevant benchmarks - maybe there is something that we have not considered

If there are none - review the options of us making our own, what criteria should it match, how would it make the tools verifiable and success evidence-based without implying to tight of a constraint (we don't want to pre-define all that the system could find and wrap into tools, as "in the wild case" would never have that) 

Maybe we can find a set of heavy scientific repos that are not trivial (not RDKit) but already have an expert-written MCP.
Or maybe we could tighten and build on the existing ToolRosella repo set.
Or we could use synthetic genertion and human assessment as steps in creation of the benchmark

Methodology questions

M1. If we do decide to build our bench that would allow us to both test Alembic and provide a way to test such systems in a thorough and stable, predictable manner - can we base our AAAI Main track submission on that, or it goes against paper themes/guidelines?

M2. How do we formulate the final problematic behind the bench we present 

M3. How we conduct our experiments.

M4. What are our RQ's
 