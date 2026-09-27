# Serving image: FastAPI + built frontend + precomputed artifacts (~270 MB).
# Build after `make data artifacts frontend`:  docker build -t goodrec .
# Run:  docker run -p 8080:8080 goodrec

FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONPATH=/app/src PATH="/app/.venv/bin:$PATH"
# Serving dependencies only (no pipeline/dev groups: no polars, implicit, anthropic).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project
COPY config/ config/
COPY src/ src/
COPY artifacts/ artifacts/
COPY --from=frontend /frontend/dist frontend/dist
EXPOSE 8080
CMD ["sh", "-c", "uvicorn goodrec.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
