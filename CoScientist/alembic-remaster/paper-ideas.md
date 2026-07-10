I want to update the core ideas of our paper. Of course they are llined out here in plain english, wording needs to be appropriate and logical within paper's structure.

Mostly keep the same structure but enforce these ideas, as well as update for current implementation

1. Abstract and Intro

in abstract - Full code with artifacts is available at [fork repo we will have just for alembic, with all the benchmark runs, tables, and without coscientist]
Demo video  is available at [demo video link]


Alembic  is is motivated by presenting an all-in-one solution for making scientific repositories more accessible 
- sometimes they are badly maintained and ill-documented, or overall hardly re-launchable.as well as not always 
mentioning thorough setup guides.

To some extent this is now solved by agentic harnesses like openclaw (and others with paper refs)
It is still executing on user's system that, with the popularity of fully-autonomous modes may pose danger 
by installing external packages and polluting global dependencies

The other motivation is reusability - if needed to conduct multiple experiments, especially in autonomous AI Scientific systems, 
where verifying the results of scientific papers the code was attached to, or reliably transforming them to tools that 
may be used for other experiments.

The user interface [screenshot]
he dashboard displays the system's work progress, live tool testing and invocation examples, as well as live access to tool calling with customizable parameters. 


2. Alembic framework

Beautiful diagram of our system (need to fina ai tools / skills that makes an actual beautiful latex-friendly 
diagrams)

Update for current bench structure and architectural decisions

The main change is that we now use part of toolmaker's architecture and build on it: our explorer and environment work as before, but coding is now split:
For the ease of coding and testing, Coder writes direct usage scripts of target workflows, after which they get tested by 
static (pytest) and invocation tests.
Then, the scripts are algorithmically wrapped in a two-layer structure: main fastmcp mcp server definition file runs venv with python 3.11, while helper scripts are ran in subprocesses with the required python version envs, avoiding package conflicts.

The venv-splitting is implemented to support running older, unmaintained repos, which we are still able to do.

We also explain our other design choices and main workflow - what was picked from the spec sheets and explianed in PLAN / design choices document, not getting too much into technical inexing, just highlighting the most interesting details that are different and imporve on ToolRosella/ToolMaker (check CoScientist/alembic/docs/TOOLMAKER_COMPARISON.md / CoScientist/alembic/docs/TOOLROSELLA_COMPARISON.md). 

- Gates are definitely a highlight, end 2 end test trying to follow the principles of manual tests from ToolMaker - to the extend it is actually possible to automatically create and test on just repo's structure. As well as adopting dockerized system with checkpoints for execution safety foth self- and to the user's system.

Toolmaker obviously does not emit MCP, and toolRosella uses a very vague 3 exposed tools success criterion

ToolRosella does not handle external keys, weights & data downloading, 



3. Experiments

It is a very complicated task to test against unknown objectives. The most information we 
can gather from repo comes from authors themselves. The system tries its best to match the 
discovered workflows in tools and keep them configurable for the agent that will know what to use it for.

We adopt ToolMaker's TMBench as there are not many benchmarks for tool/MCP creation that come
with objectives that are initially known possible, human-made tests for the tools themselves (as 
reusing the existing repository test suite does not cover the tool usage that may extend beyond covered scenarios)

from source (shorten & rephrase): 
TMBench tasks were curated in close collaboration with re-
searchers in medicine and life sciences to reflect
realistic problems in these fields, with a focus on
the medical domain (pathology, radiology, omics),
while also including some tasks from other areas
such as 3D vision, imaging, tabular data analysis,
and natural language processing to ensure broader
coverage of real-world scientific challenges.

We use glm-5.2 as our main model

We report on the runs of alembic, ToolMaker and openhands with GLM-5.2
Alembic runs its own suite, producing artefacts for TMBench scoring, ToolMaker and openhands are re-launched from toolmaker repo with a modern model. 

The toolmaker repo and bench had caveats that required fixing for consistent work, but those touched only the insignificant parts of code, like download command flags, container leakage upon run failiure, and support of models not tracked by cost in litellm and is available at https://github.com/stas1f1/ToolMaker.

TMBench comes with caveats - one of them being the huge dataset size of TCGA dataset:

  TCGA (The Cancer Genome Atlas) is a public NCI repository of cancer genomic, clinical, and imaging data — including
  whole-slide histopathology images (.svs files), hosted on the Genomic Data Commons (GDC) API. Two of the 15 TMBench
  tasks need it: stamp_extract_features and stamp_train_classification_model, which run the STAMP pipeline over
  TCGA-BRCA and TCGA-CRC slide cohorts.

  The problem: the reference CSVs ToolMaker ships (TCGA-BRCA-DX_SLIDE.csv, TCGA-CRC-DX_SLIDE.csv) list the complete
  cohorts — 1132 BRCA + 624 CRC slides — and downloading every one of them would be roughly 1.5TB. However the test 
  expects expected_num_processed_slides == 10 for the CRC test case (not 624), plus two specific named files (one CRC, 
  one BRCA) referenced by exact filename elsewhere. None of the tests check exact feature values — only shape/count/status. 
  So the full cohort was never required, just those 10+2 slides.

That is practiaclly impossible to guess beforehand, so we follow authors with turning off file downloading for these runs, 
leaving the ability to download file weights, and cloning the required data for tasks inside the container, as intended originally.

We run alembic with known target mode - it still extracts all workflows it finds, and we report metrics for 
total MCP conversion, as well as the TMBench-defined metrics presented in the initial paper

There are two task in TMBench that require the same STAMP stamp_train_classification_model and stamp_extract_features - let's add a option to run these two simultaneously, meaning explorer has to find all that is needed for both tools and coder has to also code both.

I understand that this might be tough but it would really showcase the actual Repo-to-MCP evolution of our project. If it fails, we'll just run them separately.


4. Results & Discussion

We provide comparison of our run on TMBench, ToolMalker and openhands with GLM-5.2
Alembic results are also provided in another table, featuring more of its own metrics.
Then we provide additional study for mistakes by type, token consumprtion by stage etc. 

4.1 Case study - onetask where we have good pictures / visualizations / interesting data sturcture to work with

