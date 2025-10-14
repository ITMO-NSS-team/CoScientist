# Installation guide

ChemCoScientist depends on several services: ChromaDB, embedding and reranker services, MinIO (S3), AutoML, and generative models. Each runs in its own separate Docker container.

## Installation

### Project Components Setup

1. ChromaDB, reranker service, embedding service:
    1. Clone/update the repository (`/home/chem-paper-assistant/` on the server)
    2. Run `cd infrastructure/chroma`
    3. Run `docker compose up`
2. AutoML
3. Generative models
4. ChemCoScientist app:
    1. Clone/update the repository (`/home/chem-paper-assistant/` on the server)
    2. Create a `config.env` file in the root of the project based on [example_config.env](example_config.env)
    3. Adjust the path to the volume if necessary in [docker-compose.yml](../docker/docker-compose.yml)
    4. Run `cd docker`
    5. Run `docker compose up`