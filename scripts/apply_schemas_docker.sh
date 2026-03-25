#!/usr/bin/env bash
# Apply SQL schemas via Docker 
#
# Usage:
#   ./scripts/apply_schemas_docker.sh
#   CONTAINER=adba-postgres ./scripts/apply_schemas_docker.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Load .env nếu có
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

POSTGRES_USER="${POSTGRES_USER:-adba_user}"
POSTGRES_DB="${POSTGRES_DB:-adba_db}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-adba_password}"
CONTAINER="${CONTAINER:-adba-postgres}"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "Lỗi: container '$CONTAINER' không chạy. Chạy: docker compose up -d"
  exit 1
fi

run_sql() {
  local file="$1"
  echo "→ $file"
  docker exec -i "$CONTAINER" \
    env PGPASSWORD="$POSTGRES_PASSWORD" \
    psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 < "$file"
}

run_sql "data/schemas/schema_sales.sql"
run_sql "data/schemas/schema_inventory.sql"
run_sql "data/schemas/schema_hr.sql"

echo "✓ Đã apply 3 schema (sales → inventory → hr)."
