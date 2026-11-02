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

# Litestream streams the live SQLite database to Cloud Storage (DECISIONS.md D-020). Pinned
# release, checked against its SHA-256.
FROM debian:bookworm-slim AS litestream
ARG LITESTREAM_VERSION=v0.3.13
ARG LITESTREAM_SHA256=eb75a3de5cab03875cdae9f5f539e6aedadd66607003d9b1e7a9077948818ba0
ADD https://github.com/benbjohnson/litestream/releases/download/${LITESTREAM_VERSION}/litestream-${LITESTREAM_VERSION}-linux-amd64.tar.gz /litestream.tar.gz
RUN echo "${LITESTREAM_SHA256}  /litestream.tar.gz" | sha256sum -c - \
    && tar -xzf /litestream.tar.gz -C /usr/local/bin litestream

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONPATH=/app/src PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project
RUN playwright install --with-deps chromium
COPY --from=litestream /usr/local/bin/litestream /usr/local/bin/litestream
COPY litestream.yml /etc/litestream.yml
COPY scripts/start.sh /app/start.sh
COPY src/ src/
COPY --from=frontend /frontend/dist frontend/dist
# The Timeline's data, served to approved people (api/changelog.py), never bundled. package.json is
# listed so the COPY still has a match when the Timeline wasn't built.
COPY frontend/package.json frontend/src/changelog.json* frontend/src/
EXPOSE 8080
# Plain uvicorn unless LITESTREAM_REPLICA_URL is set (production; see scripts/start.sh).
CMD ["/app/start.sh"]
