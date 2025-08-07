FROM python:3.11

WORKDIR /app

COPY . .

ENV POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && \
    apt-get install -y build-essential vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    export PATH="/root/.local/bin:$PATH" && \
    poetry --no-interaction --no-ansi --no-root install && \
    poetry run pip install --no-deps git+https://github.com/aimclub/ProtoLLM.git@main && \
    rm -rf /root/.cache/pip

CMD ["streamlit", "run", "ChemCoScientist/streamlit_app.py"]