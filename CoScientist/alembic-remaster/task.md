You are tasked with optimizing and redesigning CoScientist/alembic module.
We want to make it more competitive for the demo track paper, and perform better than the current version on benchmarks/alembic/toolmaker_subset.txt.
All of the findings, architectural choices (proposes/rejected/implemented) are in CoScientist/alembic/docs.
Right now alembic has a rather simple structure, but its code is a bit bloated as features for stability and security were often
implemented after being found an error, not thought of from the ground up, agent prompts are massive and the errors in benchmark results are inconsistent.
Also the current solution is focused around qwen/qwen3-235b-a22b-2507 model which is good but not frontier and thus some assists may be too excessive.

I want you to first make a plan how to re-design agents workflow to be more bullet-proof, maybe change the agent roles / split them / rearrange checks so that the generated code is guaranteed stable, supporting setting a list of target tools among others (as for TMBench).
You may move away from the current decisions, adapt the ones that were planned as the most promising in F/N lists but were not implemented.
I want it to be still 100% compatible with the current benchmarks/alembic/run_benchmark.py suite and docker/alembic docker infrastructure.
Take best practices from ToolRosella and ToolMaker, their repos with respective papers are in /home/stas/Documents/Github folder, as well as comparative analysis in docs.

First make a design choice document with conicse architectural choices, then plan out the works, implement and test on the whole bench (remember to clean up containers/images up until the last run, the memory may end) - when you are sure with the quality of the module - change the model to z-ai/glm-5 to perform the top-quality run.
The code needs to be concise, easy to follow and manage, files not overbloated and nicely managed.