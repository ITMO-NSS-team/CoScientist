FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY . . 

ENV UV_SYSTEM_PYTHON=1

RUN uv sync --no-dev

EXPOSE 7331

CMD ["uv", "run", "python", "-m", "server.chemical_server"]
