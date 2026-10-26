# Serving image: FastAPI + built frontend. Includes Playwright/Chromium for the custom_board
# adapter's JS-rendering fallback (confirmed necessary for at least one real source during
# development) -- this is the main reason the image is bigger than a typical FastAPI service.
# Build: docker build -t jobhunt .
# Run:   docker run -p 8080:8080 -v $(pwd)/data:/app/data jobhunt

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
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project
RUN playwright install --with-deps chromium
COPY src/ src/
COPY --from=frontend /frontend/dist frontend/dist
EXPOSE 8080
CMD ["sh", "-c", "uvicorn jobhunt.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
