FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY release-docs-gitbook-ingest.py .
COPY utils.py .
COPY chains.py .
COPY images ./images

ENTRYPOINT ["python", "release-docs-gitbook-ingest.py"]

