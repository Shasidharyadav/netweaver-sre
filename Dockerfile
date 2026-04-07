# Hugging Face Docker Spaces - NetWeaver-SRE OpenEnv
# HF requires Dockerfile at repo root, listening on port 7860

FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy all project files
COPY --chown=user . /app

# Install dependencies from pyproject.toml using pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir openenv-core openai python-dotenv numpy uvicorn fastapi

# Run the FastAPI server on port 7860 (required by HF Spaces)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
