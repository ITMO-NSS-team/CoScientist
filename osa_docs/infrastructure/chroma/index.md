
# Chroma
## Overview
The `chroma` module provides services for converting text into numerical vector representations (embeddings) and ranking the relevance of text pairs, along with integration tests to ensure the correct operation of these services and Chroma database interactions. It consists of Embedding Service, Reranker Service and Scripts for testing.

## Purpose
This module is a core component of the CoScientist project, specifically designed to enable semantic search and question answering over scientific literature. It delivers an API endpoint for embedding lists of texts and provides functionality for ranking pairs of strings based on their relevance, improving the quality of search results and recommendations. Furthermore, it validates the ability to store and retrieve data from the Chroma vector database, as well as the functionality of embedding and reranking services.
