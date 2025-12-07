import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from prefect import flow, task

from flows.analyze_current_user.active_check import fetch_data
from flows.common.db_utils import ensure_partition_exists, fast_bulk_insert
from flows.common.queries import (
    CONTRACTS_QUERY_TEMPLATE,
    PURCHASE_LOGS_QUERY_TEMPLATE,
    USER_INGESTION_QUERY,
    VOC_INGESTION_QUERY_TEMPLATE,
    WORK_LOGS_QUERY_TEMPLATE,
)
from src.dataloader import CescoCXConnection, CescoRodbConnection
from src.models.churn_prediction.predictor import ChurnPredictor
from src.models.recommendation.recommender import UserRecommender


def get_rodb_connection():
    rodb_connection = CescoRodbConnection()
    rodb_connection.connect()
    return rodb_connection


def get_bidb_connection():
    bidb_connection = CescoCXConnection()
    bidb_connection.connect()
    return bidb_connection


@task(log_prints=True, retries=1)
def ingest_users() -> pd.DataFrame:
    query = USER_INGESTION_QUERY
    rodb_connection = get_rodb_connection()
    rodb_connection.connect()
    users_data = rodb_connection.execute_query(query)
    print(f"✅ Ingested {len(users_data)} user records.")
    return users_data


@task(log_prints=True, retries=1)
def ingest_purchase_logs(customer_code: str) -> pd.DataFrame:
    query = PURCHASE_LOGS_QUERY_TEMPLATE.format(
        custcode_list=customer_code,
        inference_date=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d"),
    )

    # Create fresh connection for each query to avoid "connection is closed" errors
    bidb_connection = get_bidb_connection()
    purchase_logs_data = bidb_connection.execute_query(query)

    if len(purchase_logs_data) == 0:
        print("⚠️ No new purchase log records found.")
        return purchase_logs_data
    print(f"✅ Ingested {len(purchase_logs_data)} purchase log records.")
    return purchase_logs_data


@task(log_prints=True, retries=1)
def ingest_work_logs(customer_code: str) -> pd.DataFrame:
    query = WORK_LOGS_QUERY_TEMPLATE.format(
        custcode_list=customer_code,
        inference_date=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d"),
    )

    # Create fresh connection for each query to avoid "connection is closed" errors
    bidb_connection = get_bidb_connection()
    work_logs_data = bidb_connection.execute_query(query)

    print(f"✅ Ingested {len(work_logs_data)} work log records.")
    return work_logs_data


@task(log_prints=True, retries=1)
def ingest_contracts(customer_code: str) -> pd.DataFrame:
    query = CONTRACTS_QUERY_TEMPLATE.format(
        custcode_list=customer_code,
        inference_date=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d"),
    )

    # Create fresh connection for each query to avoid "connection is closed" errors
    rodb_connection = get_rodb_connection()
    contracts_data = rodb_connection.execute_query(query)

    print(f"✅ Ingested {len(contracts_data)} contract records.")
    return contracts_data


@task(log_prints=True, retries=1)
def ingest_users_and_contracts() -> pd.DataFrame:
    # Current time in KST (Korea Standard Time) - proper timezone handling
    now_kst = datetime.now(ZoneInfo("Asia/Seoul"))

    # Time 1 hour ago in KST
    one_hr_ago_kst = now_kst - timedelta(hours=1)
    one_hour_ago_str = one_hr_ago_kst.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Current KST: {now_kst.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"1 hour ago KST: {one_hr_ago_kst.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"Reference datetime: {one_hour_ago_str}")

    rodb_connection = get_rodb_connection()
    rodb_connection.connect()
    rodb_connection.test_connection()
    print("Ingesting VOC data...")

    formatted_query = VOC_INGESTION_QUERY_TEMPLATE.format(
        one_hour_ago_kst=one_hour_ago_str
    )
    voc_data = rodb_connection.execute_query(formatted_query)

    if voc_data.empty:
        print("⚠️ No new VOC records found in the past hour.")
        return voc_data

    print(f"✅ Ingested {len(voc_data)} VOC records.")
    return voc_data


def chunk_list(lst, chunk_size):
    """Split a list into chunks of specified size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


@task(log_prints=True, retries=1)
def predict_churn_for_chunk(
    active_customer_status_df: pd.DataFrame, work_db: pd.DataFrame
):
    churn_predictor = ChurnPredictor()
    churn_results = churn_predictor.run_inference(active_customer_status_df, work_db)
    return churn_results


@task(log_prints=True, retries=1)
def recommend_users_for_chunk(chunk_df: pd.DataFrame):
    recommender = UserRecommender()
    results = recommender.run_inference(chunk_df)
    return results


@task(log_prints=True, retries=1)
def process_and_load_chunk(chunk_df: pd.DataFrame, snapshot_date: datetime):
    chunk_df["snapshot_month"] = snapshot_date.strftime("%Y-%m-%d")
    chunk_df["purchase_logs"] = chunk_df["purchase_logs"].fillna("[]")
    chunk_df["purchase_logs"] = chunk_df["purchase_logs"].apply(
        lambda x: x if x.startswith("[") else "[]"
    )
    chunk_df["interaction_history"] = chunk_df["interaction_history"].fillna("[]")
    chunk_df["customer_info"] = chunk_df.apply(
        lambda row: json.dumps(
            {
                "고객코드": row["고객코드"],
                "고객명": row.get("고객명", ""),
                "유형대": row.get("유형대", ""),
                "유형중": row.get("유형중", ""),
                "대표자명": row.get("대표자명", ""),
                "우편번호": row.get("우편번호", ""),
                "주소1": row.get("주소1", ""),
                "주소2": row.get("주소2", ""),
                "담당부서": row.get("담당부서", ""),
                "업태": row.get("업태", ""),
                "종목": row.get("종목", ""),
                "사업자번호": row.get("사업자번호", ""),
                "성별": row.get("성별", ""),
                "국적": row.get("국적", ""),
                "신고객분류코드": row.get("신고객분류코드", ""),
                "등록일자": row.get("등록일자", ""),
            },
            ensure_ascii=False,
        ),
        axis=1,
    )
    final_df = pd.DataFrame(
        {
            "ccod": chunk_df["고객코드"],
            "snapshot_month": chunk_df["snapshot_month"],
            "user_information": chunk_df["customer_info"],
            "contract_info": chunk_df["contracts_info"],
            "purchase_logs": chunk_df["purchase_logs"],
            "interaction_history": chunk_df["interaction_history"],
        }
    )

    # Ensure the monthly partition exists for the snapshot month before bulk inserting
    ensure_partition_exists(
        target_date=snapshot_date, schema="source", table_name="user_monthly_features"
    )
    fast_bulk_insert(final_df, table_name="user_monthly_features", schema="source")
    print(
        f"✅ Processed and loaded chunk with {len(chunk_df)} records into user_monthly_features."
    )


@flow(log_prints=True)
def analyze_current_user():
    today = datetime.now()
    print(
        f"📅 Starting data ingestion and inference for date: {today.strftime('%Y-%m')}"
    )

    # prepare_database(today)
    customer_codes = ingest_users()
    print(customer_codes[:5])

    # Get the list of customer codes
    code_list = customer_codes["고객코드"].tolist()
    total_codes = len(code_list)
    chunk_size = 1000  # Process 50 customers at a time (balanced approach)

    print(f"Processing {total_codes} customer codes in chunks of {chunk_size}")
    num_active_customers = 0

    processed_count = 0

    for chunk_idx, code_chunk in enumerate(chunk_list(code_list, chunk_size)):
        chunk_start_time = time.time()

        # Format the chunk as a comma-separated string for SQL IN clause
        code_formatted = ", ".join([f"'{code}'" for code in code_chunk])

        print(f"\n📦 Processing chunk {chunk_idx + 1} ({len(code_chunk)} codes)...")

        try:
            contract_db = ingest_contracts(code_formatted)

            customer_status_df = fetch_data(contract_db)

            active_customer_status_df = customer_status_df[
                customer_status_df["최종계약상태"] == 1
            ]

            code_formatted = ", ".join(
                [
                    f"'{row['고객코드']}'"
                    for _, row in active_customer_status_df.iterrows()
                ]
            )

            work_db = ingest_work_logs(code_formatted)
            purchase_db = ingest_purchase_logs(code_formatted)

            chunk_df = active_customer_status_df.merge(
                work_db, on="고객코드", how="left"
            )
            chunk_df = chunk_df.merge(purchase_db, on="고객코드", how="left")
            process_and_load_chunk(chunk_df, today)
            processed_count += len(code_chunk)

            chunk_end_time = time.time()
            chunk_duration = chunk_end_time - chunk_start_time

            num_active_customers += len(active_customer_status_df)
            print(f"Current aggregated active customers: {num_active_customers}")

            chunk_churn_results = predict_churn_for_chunk(
                active_customer_status_df, work_db
            )
            print(
                f"Churn Prediction Results for Chunk {chunk_idx + 1}:\n{chunk_churn_results}"
            )

            chunk_recommendation_results = recommend_users_for_chunk(chunk_df)
            print(
                f"Recommendation Results for Chunk {chunk_idx + 1}:\n{chunk_recommendation_results}"
            )

            print(f"✅ Chunk {chunk_idx + 1} completed in {chunk_duration:.2f} seconds")
            print(f"   - Processed {len(code_chunk)} codes")
            print(f"   - Retrieved {len(contract_db)} contract records")
            print(
                f"   - Total progress: {processed_count}/{total_codes} ({processed_count/total_codes*100:.1f}%)"
            )

            # Optional: Add a small delay between chunks to avoid overwhelming the database
            time.sleep(0.1)

        except Exception as e:
            print(f"❌ Error processing chunk {chunk_idx + 1}: {str(e)}")
            break

    recommender = UserRecommender()
    print("All chunks processed. Finalizing recommendations...")
    recommender.update_cluster_info(snapshot_month=today.strftime("%Y_%m"))


if __name__ == "__main__":
    analyze_current_user()
