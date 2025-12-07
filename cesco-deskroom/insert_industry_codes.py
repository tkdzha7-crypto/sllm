import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import create_engine


def main():
    deskroom_core_db_user = os.getenv("DB_USER", "cesco_admin")
    deskroom_core_db_pass = os.getenv("DB_PASS", "Cesco_1588")
    deskroom_core_db_host = os.getenv("DB_HOST", "localhost")
    deskroom_core_db_name = os.getenv("DB_NAME", "deskroom_core")
    now_kst = datetime.now(ZoneInfo("Asia/Seoul"))

    deskroom_core_db_url = f"postgresql://{deskroom_core_db_user}:{deskroom_core_db_pass}@{deskroom_core_db_host}:5432/{deskroom_core_db_name}"
    deskroom_core_engine = create_engine(deskroom_core_db_url)

    voc_category_table = pd.read_csv("./local_files/bz_code_10_sm_converted.csv")

    # Select only the columns that exist in the database table (excluding 등록일시)
    formatted_data = voc_category_table[["code", "항목명"]].copy()
    formatted_data.columns = ["code", "name"]

    # Add created_at column with current KST time
    formatted_data["created_at"] = now_kst

    # Add updated_at column by copying created_at values
    formatted_data["updated_at"] = formatted_data["created_at"].copy()

    # Insert data into the database
    formatted_data.to_sql(
        "industry_codes",
        deskroom_core_engine,
        if_exists="append",
        index=False,
        schema="source",
    )
    print(f"✅ Inserted {len(formatted_data)} records into voc_category table.")


if __name__ == "__main__":
    main()
