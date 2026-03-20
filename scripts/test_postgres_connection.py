import os
from urllib.parse import urlparse

import psycopg2


def build_dsn() -> str:
    postgres_url = os.getenv("POSTGRES_URL")
    if postgres_url:
        parsed = urlparse(postgres_url)
        if parsed.scheme not in {"postgres", "postgresql"}:
            raise ValueError("POSTGRES_URL must use postgres/postgresql scheme")
        return postgres_url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "adba_db")
    user = os.getenv("POSTGRES_USER", "adba_user")
    password = os.getenv("POSTGRES_PASSWORD", "adba_password")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def main() -> None:
    dsn = build_dsn()
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            cur.execute("SELECT current_database(), current_user;")
            db_name, db_user = cur.fetchone()
        print("PostgreSQL connection successful")
        print(f"- Database: {db_name}")
        print(f"- User: {db_user}")
        print(f"- Version: {version}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
