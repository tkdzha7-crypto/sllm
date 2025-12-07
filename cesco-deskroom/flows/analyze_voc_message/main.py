from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from prefect import flow, task
from sqlalchemy import text

from flows.analyze_voc_message.processor import VoCAnalyzer
from flows.common.db_utils import get_engine
from src.dataloader import CescoRodbConnection


@task(log_prints=True)
def analyze_message_data(
    message_data: pd.DataFrame,
    run_in_batch: bool = True,
) -> pd.DataFrame:
    deskroom_core_engine = get_engine()
    voc_analyzer = VoCAnalyzer(db_connection=deskroom_core_engine)
    voc_inferred_results = []
    if run_in_batch:
        ## Chunk messages into batches for processing
        batch_size = 8
        for i in range(0, len(message_data), batch_size):
            batch_df = message_data.iloc[i : i + batch_size]
            batch_results = voc_analyzer.analyze_message_in_batch(batch_df)
            voc_inferred_results.extend(batch_results)
        return pd.DataFrame(voc_inferred_results)
    else:
        for sample in message_data.itertuples():
            sample_analysis = voc_analyzer.analyze_message(sample)
            voc_inferred_results.append(sample_analysis)
        return pd.DataFrame(voc_inferred_results)


@task(log_prints=True, retries=1)
def ingest_voc_data(
    start_time: str = "",
    end_time: str = "",
):
    rodb_connection = CescoRodbConnection()
    rodb_connection.connect()

    deskroom_core_engine = get_engine()
    if not start_time == "" and not end_time == "":
        print(f"Using provided start_time: {start_time}, end_time: {end_time}")
    else:
        # Current time in KST (Korea Standard Time) - proper timezone handling
        now_kst = datetime.now(ZoneInfo("Asia/Seoul"))

        # Time 1 hour ago in KST
        one_hr_ago_kst = now_kst - timedelta(hours=1)
        one_hour_ago_str = one_hr_ago_kst.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Current KST: {now_kst.strftime('%Y-%m-%d %H:%M %Z')}")
        print(f"1 hour ago KST: {one_hr_ago_kst.strftime('%Y-%m-%d %H:%M %Z')}")
        print(f"Reference datetime: {one_hour_ago_str}")

        start_time = one_hour_ago_str
        end_time = now_kst.strftime("%Y-%m-%d %H:%M:%S")
    rodb_connection.test_connection()
    print("Ingesting VOC data...")

    query = """
SELECT
	RCNO as 접수번호,
	CONVERT(
		DATETIME,
		CONCAT(
			RCDT,
			' ',
			STUFF(RCTM, 3,0,':'),
			':00'
			), 120
		) as 접수일시,
	RCDT as 접수일자,
	RCTM as 접수시간,
    CONVERT(DATETIME ,INDT, 23) as 등록일시,
	CTCD as 고객코드,
	CMMO as 접수내용
FROM CESCOEIS.dbo.TB_VOC_Master
where CONVERT(DATETIME, INDT, 23) >= '{start_time}'
    AND CONVERT(DATETIME, INDT, 23) < '{end_time}'
"""
    formatted_query = query.format(start_time=start_time, end_time=end_time)
    voc_data = rodb_connection.execute_query(formatted_query)
    print(f"✅ Ingested {len(voc_data)} VOC records.")

    formatted_data = voc_data.rename(
        columns={
            "접수번호": "rcno",
            "등록일시": "received_at",
            "고객코드": "ccod",  # Note: column name is 'ccod' not 'ccode' in database
            "접수내용": "content",
        }
    )

    formatted_data = formatted_data[["rcno", "received_at", "ccod", "content"]].copy()

    # Add created_at column with current KST time
    formatted_data["created_at"] = now_kst

    # Add updated_at column by copying created_at values
    formatted_data["updated_at"] = formatted_data["created_at"].copy()

    print(formatted_data.head())
    # Insert data and get the msg_ids back
    formatted_data.to_sql(
        name="message",
        con=deskroom_core_engine,
        if_exists="append",
        index=False,
        schema="source",
    )
    print("✅ VOC data inserted into message table.")

    rcno_list = "', '".join(formatted_data["rcno"].astype(str))
    query_ids = f"""
    SELECT id, rcno FROM source.message
    WHERE rcno IN ('{rcno_list}')
    ORDER BY id DESC
    """

    with deskroom_core_engine.connect() as conn:
        result = conn.execute(text(query_ids))
        id_mapping = {row[1]: row[0] for row in result.fetchall()}  # {rcno: msg_id}

    # Add msg_id to formatted_data
    formatted_data["msg_id"] = formatted_data["rcno"].map(id_mapping)
    return formatted_data


@task(log_prints=True, retries=1, cache_policy=None)
def store_aggregated_results(db_engine, aggregated_df):
    """
    Store aggregated VOC results in the user_voc_activity table using UPSERT strategy.

    Args:
        db_engine: SQLAlchemy database engine
        aggregated_df: DataFrame with aggregated VOC data
    """
    if aggregated_df.empty:
        print("⚠️ No aggregated data to store")
        return

    try:
        # Transform the DataFrame to match the user_voc_acvity table schema
        storage_df = aggregated_df.rename(
            columns={
                "aggregate_date": "aggregate_date",
                "ccod": "ccod",  # Note: keeping lowercase to match your table
                "category_id": "category_code",
                "name": "category_name",
                "재인입_당일": "recontact_agg_day",
                "재인입_전일_24h": "recontact_past_24h",
                "재인입_3일": "recontact_past_3d",
                "재인입_7일": "recontact_past_7d",
                "재인입_30일": "recontact_past_30d",
            }
        )

        # Add timestamps
        from datetime import datetime
        from zoneinfo import ZoneInfo

        now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
        storage_df["created_at"] = now_kst
        storage_df["updated_at"] = now_kst

        # Select only the columns that exist in the target table
        storage_df = storage_df[
            [
                "aggregate_date",
                "ccod",
                "category_code",
                "category_name",
                "recontact_agg_day",
                "recontact_past_24h",
                "recontact_past_3d",
                "recontact_past_7d",
                "recontact_past_30d",
                "created_at",
                "updated_at",
            ]
        ]

        print(
            f"🔄 Storing {len(storage_df)} aggregated records in user_voc_activity table..."
        )

        # Use PostgreSQL UPSERT with ON CONFLICT for better performance
        with db_engine.connect() as conn:
            trans = conn.begin()
            try:
                # Prepare UPSERT query using ON CONFLICT
                upsert_query = text("""
                    INSERT INTO analytics.user_voc_activity (
                        aggregate_date, ccod, category_code, category_name,
                        recontact_agg_day, recontact_past_24h, recontact_past_3d,
                        recontact_past_7d, recontact_past_30d, created_at, updated_at
                    ) VALUES (
                        :aggregate_date, :ccod, :category_code, :category_name,
                        :recontact_agg_day, :recontact_past_24h, :recontact_past_3d,
                        :recontact_past_7d, :recontact_past_30d, :created_at, :updated_at
                    )
                    ON CONFLICT (aggregate_date, ccod, category_code)
                    DO UPDATE SET
                        category_name = EXCLUDED.category_name,
                        recontact_agg_day = EXCLUDED.recontact_agg_day,
                        recontact_past_24h = EXCLUDED.recontact_past_24h,
                        recontact_past_3d = EXCLUDED.recontact_past_3d,
                        recontact_past_7d = EXCLUDED.recontact_past_7d,
                        recontact_past_30d = EXCLUDED.recontact_past_30d,
                        updated_at = EXCLUDED.updated_at
                """)

                # Execute batch insert/update
                records_data = storage_df.to_dict("records")
                conn.execute(upsert_query, records_data)

                trans.commit()
                print(f"✅ Successfully upserted {len(storage_df)} aggregated records")

            except Exception as e:
                trans.rollback()
                print(f"❌ Transaction rolled back due to error: {str(e)}")
                raise e

    except Exception as e:
        print(f"❌ Error storing aggregated results: {str(e)}")
        raise


@task(log_prints=True, retries=1, cache_policy=None)
def reload_voc_activity_table(db_connection):
    """
    Truncate and reload the user_voc_activity table with claims-only data.
    This task aggregates historical VOC data for all dates in the past 35 days.
    """
    reload_query = """
WITH relevant_data AS (
    SELECT
        m.id,
        m.CCOD,
        m.created_at,
        DATE(m.created_at) as msg_date,
        u.category_id
    FROM source.message m
    INNER JOIN analytics.voc_message_category c
        ON m.id = c.msg_id
    CROSS JOIN LATERAL unnest(ARRAY[
        c.detail_category_1_code,
        c.detail_category_2_code,
        c.detail_category_3_code,
        c.detail_category_4_code,
        c.detail_category_5_code
    ]) AS u(category_id)
    WHERE
        m.created_at >= CURRENT_DATE - INTERVAL '35 days'
        AND u.category_id IS NOT NULL
        AND c.is_claim = 1
),
daily_counts AS (
    SELECT
        msg_date,
        CCOD,
        category_id,
        COUNT(*) AS daily_cnt
    FROM relevant_data
    GROUP BY msg_date, CCOD, category_id
),
calculated_windows AS (
    SELECT
        msg_date as aggregate_date,
        CCOD,
        category_id,
        daily_cnt,
        -- 당일 건수 (excluding current row, so subtract 1 if we want "재인입")
        daily_cnt - 1 AS cnt_today,
        -- 전일 포함 (yesterday + today)
        COALESCE(SUM(daily_cnt) OVER (
            PARTITION BY CCOD, category_id
            ORDER BY msg_date
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ), 0) - 1 AS cnt_1d,
        -- 과거 3일 포함 (including today)
        COALESCE(SUM(daily_cnt) OVER (
            PARTITION BY CCOD, category_id
            ORDER BY msg_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ), 0) - 1 AS cnt_3d,
        -- 과거 7일 포함 (including today)
        COALESCE(SUM(daily_cnt) OVER (
            PARTITION BY CCOD, category_id
            ORDER BY msg_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 0) - 1 AS cnt_7d,
        -- 과거 30일 포함 (including today)
        COALESCE(SUM(daily_cnt) OVER (
            PARTITION BY CCOD, category_id
            ORDER BY msg_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ), 0) - 1 AS cnt_30d
    FROM daily_counts
)
SELECT
    aggregate_date,
    CCOD as ccod,
    category_id,
    vc.name,
    cnt_today AS 재인입_당일,
    cnt_1d    AS 재인입_전일_24h,
    cnt_3d    AS 재인입_3일,
    cnt_7d    AS 재인입_7일,
    cnt_30d   AS 재인입_30일
FROM calculated_windows
LEFT JOIN source.voc_category vc ON category_id = vc.voc_id;
"""

    try:
        with db_connection.connect() as conn:
            # Step 1: Truncate the table
            print("🗑️ Truncating analytics.user_voc_activity table...")
            conn.execute(text("TRUNCATE TABLE analytics.user_voc_activity;"))
            conn.commit()
            print("✅ Table truncated successfully")

            # Step 2: Execute the aggregation query
            print("🔄 Executing full VOC aggregation query (claims only)...")
            result_df = pd.read_sql(text(reload_query), conn)

        if result_df.empty:
            print("⚠️ No VOC activities found for reload (empty result set)")
            return result_df

        print(f"✅ Retrieved {len(result_df)} records for reload")

        # Log summary statistics
        unique_ccods = result_df["ccod"].nunique() if "ccod" in result_df.columns else 0
        unique_categories = (
            result_df["category_id"].nunique()
            if "category_id" in result_df.columns
            else 0
        )
        unique_dates = (
            result_df["aggregate_date"].nunique()
            if "aggregate_date" in result_df.columns
            else 0
        )
        print(
            f"📊 Summary: {unique_ccods} unique customers, {unique_categories} unique categories, {unique_dates} unique dates"
        )

        # Step 3: Store results in user_voc_activity table
        store_aggregated_results(db_connection, result_df)

        return result_df

    except Exception as e:
        print(f"❌ Error during VOC table reload: {str(e)}")
        raise


@task(log_prints=True, retries=1, cache_policy=None)
def aggregate_voc_activities(db_connection):
    aggregate_query = """
WITH relevant_data AS (
    SELECT
        m.id,
        m.CCOD,
        m.created_at,
        DATE(m.created_at) as msg_date,
        u.category_id
    FROM source.message m
    INNER JOIN analytics.voc_message_category c
        ON m.id = c.msg_id
    CROSS JOIN LATERAL unnest(ARRAY[
        c.detail_category_1_code,
        c.detail_category_2_code,
        c.detail_category_3_code,
        c.detail_category_4_code,
        c.detail_category_5_code
    ]) AS u(category_id)
    WHERE
        m.created_at >= CURRENT_DATE - INTERVAL '35 days'
        AND u.category_id IS NOT NULL
        AND c.is_claim = 1
),
daily_counts AS (
    SELECT
        msg_date,
        CCOD,
        category_id,
        COUNT(*) AS daily_cnt
    FROM relevant_data
    GROUP BY msg_date, CCOD, category_id
),
calculated_windows AS (
    SELECT
        msg_date as aggregate_date,
        CCOD,
        category_id,
        daily_cnt,
        -- 당일 건수 (excluding current row, so subtract 1 if we want "재인입")
        daily_cnt - 1 AS cnt_today,
        -- 전일 포함 (yesterday + today)
        COALESCE(SUM(daily_cnt) OVER (
            PARTITION BY CCOD, category_id
            ORDER BY msg_date
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ), 0) - 1 AS cnt_1d,
        -- 과거 3일 포함 (including today)
        COALESCE(SUM(daily_cnt) OVER (
            PARTITION BY CCOD, category_id
            ORDER BY msg_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ), 0) - 1 AS cnt_3d,
        -- 과거 7일 포함 (including today)
        COALESCE(SUM(daily_cnt) OVER (
            PARTITION BY CCOD, category_id
            ORDER BY msg_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 0) - 1 AS cnt_7d,
        -- 과거 30일 포함 (including today)
        COALESCE(SUM(daily_cnt) OVER (
            PARTITION BY CCOD, category_id
            ORDER BY msg_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ), 0) - 1 AS cnt_30d
    FROM daily_counts
)
SELECT
    aggregate_date,
    CCOD as ccod,
    category_id,
    vc.name,
    cnt_today AS 재인입_당일,
    cnt_1d    AS 재인입_전일_24h,
    cnt_3d    AS 재인입_3일,
    cnt_7d    AS 재인입_7일,
    cnt_30d   AS 재인입_30일
FROM calculated_windows
LEFT JOIN source.voc_category vc ON category_id = vc.voc_id
WHERE aggregate_date = CURRENT_DATE;
"""

    try:
        with db_connection.connect() as conn:
            print("🔄 Executing VOC aggregation query...")
            result_df = pd.read_sql(text(aggregate_query), conn)

        if result_df.empty:
            print("⚠️ No VOC activities found for aggregation (empty result set)")
            return result_df

        print(f"✅ Successfully aggregated VOC activities: {len(result_df)} records")

        # Log summary statistics for monitoring
        unique_ccods = result_df["ccod"].nunique() if "ccod" in result_df.columns else 0
        unique_categories = (
            result_df["category_id"].nunique()
            if "category_id" in result_df.columns
            else 0
        )
        print(
            f"📊 Summary: {unique_ccods} unique customers, {unique_categories} unique categories"
        )

        # Store results in user_voc_acvity table
        if not result_df.empty:
            store_aggregated_results(db_connection, result_df)

        return result_df

    except Exception as e:
        print(f"❌ Error during VOC aggregation: {str(e)}")
        print(f"📋 Query length: {len(aggregate_query)} characters")
        raise


@flow(log_prints=True)
def analyze_voc(
    start_time: str = "",
    end_time: str = "",
    run_in_batch: bool = True,
):
    deskroom_core_engine = get_engine()
    message_data = ingest_voc_data(
        start_time=start_time,
        end_time=end_time,
    )
    analyzed_data = analyze_message_data(message_data, run_in_batch=run_in_batch)
    analyzed_data.to_sql(
        name="voc_message_category",
        con=deskroom_core_engine,
        if_exists="append",
        index=False,
        schema="analytics",
    )
    aggregate_voc_activities(deskroom_core_engine)


@flow(log_prints=True)
def reload_voc_activity():
    """
    Flow to truncate and reload the user_voc_activity table with claims-only data.
    Run this once to migrate existing data to the new claims-only logic.
    """
    deskroom_core_engine = get_engine()
    reload_voc_activity_table(deskroom_core_engine)
    print("🎉 VOC activity table reload complete!")


if __name__ == "__main__":
    analyze_voc(start_time="2023-01-01", end_time="2023-01-31", run_in_batch=True)
