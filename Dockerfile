# Hugging Face Docker Spaces - NetWeaver-SRE OpenEnv
# HF requires Dockerfile at repo root, listening on port 7860

FROM python:3.11-slim

RUN apt-get update && apt-get install -y socat && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy all project files
COPY --chown=user . /app

# Install dependencies from pyproject.toml using pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir openenv-core openai python-dotenv numpy uvicorn fastapi

# Copy start script explicitly and ensure it's executable
COPY --chown=user start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Run the FastAPI server via the start script
ENV PORT=7860
EXPOSE 7860 8000
CMD ["/app/start.sh"]
