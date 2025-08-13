

# chem_system_example

## Overview

The `chem_system_example` module is the core of the CoScientist project, designed for analyzing scientific papers, specifically in chemistry, and providing answers to user questions based on their content. It comprises several submodules responsible for user interaction, execution of complex problem-solving processes, prompt management, structured data representation, and chemical tool functionalities. It uses a vector database (ChromaDB) for efficient information retrieval.

## Purpose

This module serves as the primary engine for the CoScientist project, enabling users to query, analyze, and receive insights from scientific literature in the chemistry domain. The module orchestrates the entire process, from user input and question formulation through to response generation and result summarization, utilizing large language models (LLMs) and external services.

**Submodules:**

*   **Frontend:**  This module handles the user interface and interaction logic, managing chat history, user input, session state, and file uploads. It supports both English and Russian languages and provides example queries. It's responsible for providing a user-friendly way to interact with the system.
*   **Graph:** This module implements the core logic for orchestrating the execution of a planning and response generation pipeline. It manages communication with remote services, handles language translation, interacts with LLMs, and incorporates error handling and retry mechanisms. It facilitates web searches and result summarization for scientific problem-solving.
*   **Prompts:** This module is currently a placeholder designated for housing the prompts used for interacting with the LLMs, that will be used for querying and analyzing scientific papers and structuring user questions. It represents the interface through which users communicate their information needs to the CoScientist system.
*   **Pydantic Models:** This module defines the data structures used to represent responses, plans, actions, worker definitions, chats and translations within the CoScientist project, facilitating data consistency and clarity throughout the workflow.
*   **CHEMIC tools:** This module provides a suite of tools for chemical property prediction, molecule manipulation, and representation. It encompasses functionalities for converting molecule identifiers, calculating molecular properties, visualizing molecules, and generating new molecules. It provides the foundational chemistry toolbox for the project.