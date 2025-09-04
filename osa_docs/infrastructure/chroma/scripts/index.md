
# Scripts
## Overview
This module contains integration tests for key services within the CoScientist project. These tests verify the functionality of the Chroma database interaction, the embedding service, and the reranker service.

## Purpose
The purpose of this module is to ensure the reliability and correct operation of core components responsible for data storage, text embedding generation, and relevance ranking. Specifically, it validates the ability to store and retrieve data from the Chroma vector database, to obtain vector embeddings from text using the embedding service, and to re-rank results based on relevance using the reranker service. These functionalities are crucial for the question-answering system’s overall performance and accuracy when dealing with scientific literature.