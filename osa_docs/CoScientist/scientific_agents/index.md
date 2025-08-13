
# Scientific Agents
## Overview
The `scientific_agents` module is dedicated to the implementation of agents capable of generating code to execute specific tasks within a larger scientific processing pipeline. It focuses on dynamically producing executable code based on given instructions and configurations.

## Purpose
This module provides functionalities to initialize and run a code generation agent. The agent leverages either Groq or OpenAI models, receiving a task description and a list of available libraries. The agent then generates code which is structured as a `Command` object, containing necessary calls and a record of past steps, for subsequent execution. This module is specifically designed to automate code creation as a component within the CoScientist project, supporting automated workflows and task execution related to scientific analysis.