# The API and demo page, as one container. See docs/deploy.md.
#
# CPU-only torch: the embedding and reranking models are small and run once per retrieval, so
# a GPU buys nothing here and the CUDA wheels would add ~2.5GB to the image. Both models are
# downloaded at build time, so a cold start never waits on Hugging Face and the running
# container needs no outbound access except to Postgres and, in live mode, the model API.

FROM python:3.12.7-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf-cache \
    HF_HUB_OFFLINE=0 \
    FV_REPLAY=true

WORKDIR /app

# Pinned to the versions the local venv runs and the tests pass on. transformers and torch
# must move together: a newer transformers against an older torch fails at import.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
 && pip install transformers==5.14.1 sentence-transformers==5.6.1

# Dependencies first, so a source change does not reinstall them.
COPY pyproject.toml README.md ./
RUN mkdir -p src/financevault && touch src/financevault/__init__.py \
 && pip install ".[embed]" \
 && pip uninstall -y financevault

RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1

COPY src ./src
RUN pip install --no-deps .

# Read at runtime by relative path: seeded-error mutations, committed results, figures.
COPY bench ./bench
COPY eval/__init__.py ./eval/__init__.py
COPY eval/results ./eval/results
COPY report/figures ./report/figures

# Nothing under /app is written at runtime, so the files stay root-owned and read-only to fv.
RUN useradd --create-home --uid 10001 fv
USER fv

EXPOSE 8080
# One worker: run streams live in this process's memory (see api/streams.py), so a second
# worker would serve event requests for runs it never started.
CMD ["uvicorn", "financevault.api.app:app", "--host", "0.0.0.0", "--port", "8080", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--workers", "1"]
