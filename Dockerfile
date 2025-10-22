FROM python:3.11-slim
WORKDIR /app


# Pyserini (BM25/Lucene) требует Java
RUN apt-get update && apt-get install -y --no-install-recommends \
openjdk-17-jre-headless \
&& rm -rf /var/lib/apt/lists/*


# зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# исходники
COPY . .


# переменные по умолчанию (можно переопределить через compose/.env)
ENV QDRANT_URL=http://qdrant:6333 \
EMBEDDING_MODEL=BAAI/bge-m3 \
RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
MODEL_ID=llama3.1:8b \
LLM_BASE_URL=http://host.docker.internal:11434

