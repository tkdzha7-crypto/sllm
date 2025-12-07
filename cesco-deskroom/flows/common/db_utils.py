import io
import os
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine, text

deskroom_core_db_user = os.getenv("DB_USER", "cesco_admin")
deskroom_core_db_pass = os.getenv("DB_PASS", "Cesco_1588")
deskroom_core_db_host = os.getenv("DB_HOST", "localhost")
deskroom_core_db_name = os.getenv("DB_NAME", "deskroom_core")
deskroom_core_db_url = f"postgresql://{deskroom_core_db_user}:{deskroom_core_db_pass}@{deskroom_core_db_host}:5432/{deskroom_core_db_name}"

# Create a single shared engine with proper connection pooling
deskroom_core_engine = create_engine(
    deskroom_core_db_url,
    pool_size=5,  # Maximum number of permanent connections
    max_overflow=10,  # Maximum number of temporary connections
    pool_pre_ping=True,  # Verify connections before using them
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=False,
)


def get_engine():
    """Return the shared SQLAlchemy engine."""
    return deskroom_core_engine


def ensure_partition_exists(
    target_date: datetime,
    schema: str = "source",
    table_name: str = "user_monthly_features",
):
    """Ensure that the monthly partition for the given date exists in the risk_signals table."""
    engine = get_engine()
    start_date = target_date.replace(day=1)
    next_month = (start_date + timedelta(days=32)).replace(day=1)

    partition_name = f"{table_name}_{start_date.strftime('%Y_%m')}"
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = next_month.strftime("%Y-%m-%d")

    sql = text(f"""
    CREATE TABLE IF NOT EXISTS {schema}.{partition_name}
        PARTITION OF {schema}.{table_name}
        FOR VALUES FROM ('{start_str}') TO ('{end_str}');
    """)

    with engine.connect() as conn:
        conn.execute(sql)
        conn.commit()
        print(f" Partition checked/created for {schema}.{partition_name} ")


def fast_bulk_insert(df: pd.DataFrame, table_name: str, schema: str = "source"):
    """Perform a fast bulk insert of a DataFrame into the specified table."""
    engine = get_engine()
    buffer = io.StringIO()

    df.to_csv(buffer, sep="\t", index=False, header=False, na_rep="")
    buffer.seek(0)

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cursor:
            columns_list = ", ".join(df.columns)
            sql = f"""
            COPY {schema}.{table_name} ({columns_list})
            FROM STDIN
            WITH (FORMAT CSV, DELIMITER '\t', NULL '');
            """

            cursor.copy_expert(sql=sql, file=buffer)
        raw_conn.commit()
        print(
            f" Bulk insert completed into {schema}.{table_name} with {len(df)} records. "
        )

    except Exception as e:
        raw_conn.rollback()
        print(f" Error during bulk insert into {schema}.{table_name}: {e} ")
    finally:
        raw_conn.close()
