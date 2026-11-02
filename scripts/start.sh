#!/bin/sh
# Container entrypoint (see Dockerfile and DECISIONS.md D-020).
#
# With LITESTREAM_REPLICA_URL set (production), the database lives on the server's own disk at
# JOBHUNT_DB_PATH and Litestream keeps a continuous copy of it in Cloud Storage:
#   1. restore the latest copy, if there is one;
#   2. the very first time (no copy yet), seed from the database the app used before Litestream
#      (JOBHUNT_SEED_DB, the old file on the bucket mount), and leave a marker so that is never
#      repeated: if the copy ever goes missing after that, refuse to start rather than quietly
#      bringing back that old file or starting empty (which would look like a fresh install);
#   3. run the app under `litestream replicate -exec`, which streams changes while it runs and
#      flushes the last ones when Cloud Run stops the server.
# Without it (local Docker runs), this just starts the app.
set -eu

PORT="${PORT:-8080}"
APP="uvicorn jobhunt.api.main:app --host 0.0.0.0 --port $PORT"

if [ -z "${LITESTREAM_REPLICA_URL:-}" ]; then
  exec $APP
fi

: "${JOBHUNT_DB_PATH:?set JOBHUNT_DB_PATH to the database path on local disk}"
CONFIG="${LITESTREAM_CONFIG:-/etc/litestream.yml}"
MARKER="${JOBHUNT_DATA_DIR:-/data}/litestream-seeded"

mkdir -p "$(dirname "$JOBHUNT_DB_PATH")"
litestream restore -config "$CONFIG" -if-db-not-exists -if-replica-exists "$JOBHUNT_DB_PATH"

if [ ! -f "$JOBHUNT_DB_PATH" ]; then
  if [ -f "$MARKER" ]; then
    echo "start.sh: no Litestream copy found at $LITESTREAM_REPLICA_URL, but $MARKER says one was" >&2
    echo "start.sh: made before. Refusing to start empty. Restore from backups/ (see CLAUDE.md)," >&2
    echo "start.sh: or delete the marker to seed again from ${JOBHUNT_SEED_DB:-the old database}." >&2
    exit 1
  fi
  if [ -n "${JOBHUNT_SEED_DB:-}" ] && [ -f "$JOBHUNT_SEED_DB" ]; then
    echo "start.sh: first start with Litestream: seeding from $JOBHUNT_SEED_DB"
    python -m jobhunt.cli.seed_db "$JOBHUNT_SEED_DB" "$JOBHUNT_DB_PATH"
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$MARKER"
fi

exec litestream replicate -config "$CONFIG" -exec "$APP"
