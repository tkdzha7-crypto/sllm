import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.blocks.system import Secret

from flows.analyze_current_user.active_check import fetch_data
from flows.common.db_utils import ensure_partition_exists, fast_bulk_insert, get_engine
from src.dataloader import CescoCXConnection, CescoRodbConnection
from src.models.churn_prediction.predictor import ChurnPredictor
from src.models.recommendation.recommender import UserRecommender

# Cache for secrets to avoid repeated API calls
_query_cache: dict[str, str] = {}


def load_queries_from_secret() -> dict[str, str]:
    QUERY_NAMES = [
        "contracts-query",
        "csi-survey-query",
        "kodata-query",
        "pswr-detail-query",
        "pswr-query",
        "purchase-log-query",
        "towr-query",
        "user-ingestion-query",
        "work-detail-query",
    ]
    queries = {}
    for query_name in QUERY_NAMES:
        secret_block = Secret.load(query_name)
        queries[query_name] = secret_block.get()

    return queries


def load_query_from_secret(query_name: str) -> str:
    """Load query from Prefect secret with caching to avoid repeated API calls."""
    if query_name not in _query_cache:
        secret_block = Secret.load(query_name)
        _query_cache[query_name] = secret_block.get()
    return _query_cache[query_name]


def get_rodb_connection():
    rodb_connection = CescoRodbConnection()
    rodb_connection.connect()
    return rodb_connection


def get_bidb_connection():
    bidb_connection = CescoCXConnection()
    bidb_connection.connect()
    return bidb_connection


def get_processed_customers():
    db_engine = get_engine()
    query = "SELECT DISTINCT ccod FROM source.user_monthly_features"
    processed_customers = pd.read_sql(query, db_engine)
    return set(processed_customers["ccod"].tolist())


@task(log_prints=True, retries=1)
def ingest_users() -> pd.DataFrame:
    query = load_query_from_secret("user-ingestion-query")
    rodb_connection = get_rodb_connection()
    users_data = rodb_connection.execute_query(query)
    print(f"✅ Ingested {len(users_data)} user records.")
    return users_data


@task(log_prints=True, retries=1)
def ingest_purchase_logs(customer_code: str) -> pd.DataFrame:
    query = load_query_from_secret("purchase-log-query").format(
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
def ingest_towr_data(customer_code: str) -> pd.DataFrame:
    query = load_query_from_secret("towr-query").format(
        code_list=customer_code,
        inference_date=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d"),
    )

    bidb_connection = get_bidb_connection()
    towr_data = bidb_connection.execute_query(query)

    print(f"✅ Ingested {len(towr_data)} TOWR records.")
    return towr_data


@task(log_prints=True, retries=1)
def ingest_csi_survey_data(customer_code: str) -> pd.DataFrame:
    query = load_query_from_secret("csi-survey-query").format(
        code_list=customer_code,
        inference_date=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d"),
    )

    bidb_connection = get_bidb_connection()
    csi_survey_data = bidb_connection.execute_query(query)

    print(f"✅ Ingested {len(csi_survey_data)} CSI Survey records.")
    return csi_survey_data


@task(log_prints=True, retries=1)
def ingest_detail_work_data(customer_code: str) -> pd.DataFrame:
    query = load_query_from_secret("work-detail-query").format(
        code_list=customer_code,
        inference_date=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d"),
    )

    bidb_connection = get_bidb_connection()
    detail_work_data = bidb_connection.execute_query(query)

    print(f"✅ Ingested {len(detail_work_data)} Detail Work records.")
    return detail_work_data


@task(log_prints=True, retries=1)
def ingest_pswr_data(customer_code: str) -> pd.DataFrame:
    query = load_query_from_secret("pswr-query").format(
        code_list=customer_code,
        inference_date=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d"),
    )

    bidb_connection = get_bidb_connection()
    pwsr_data = bidb_connection.execute_query(query)

    print(f"✅ Ingested {len(pwsr_data)} PWSR records.")
    return pwsr_data


@task(log_prints=True, retries=1)
def ingest_pswr_detail_data(customer_code: str) -> pd.DataFrame:
    query = load_query_from_secret("pswr-detail-query").format(
        code_list=customer_code,
        inference_date=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d"),
    )

    bidb_connection = get_bidb_connection()
    pwsr_detail_data = bidb_connection.execute_query(query)

    print(f"✅ Ingested {len(pwsr_detail_data)} PWSR Detail records.")
    return pwsr_detail_data


def pack_children(parent_df, child_df, parent_key, child_key, target_col_name):
    """
    Groups child rows into a list of dictionaries and merges to parent.
    Simulates: (SELECT ... FOR JSON PATH) AS target_col_name
    """
    if child_df.empty:
        parent_df[target_col_name] = np.empty((len(parent_df), 0)).tolist()
        return parent_df

    # 1. Group by the key and convert remaining columns to List of Dicts
    # Note: We drop the joining key from the dict to match SQL behavior
    grouped = (
        child_df.groupby(child_key)
        .apply(
            lambda x: x.drop(columns=[child_key], errors="ignore").to_dict(
                orient="records"
            )
        )
        .reset_index(name=target_col_name)
    )

    # 2. Merge back to parent
    merged = parent_df.merge(
        grouped, left_on=parent_key, right_on=child_key, how="left"
    )

    # 3. Handle NULLs (Towers with no surveys should be empty list [], not NaN)
    # If you prefer NULL like SQL, remove the apply line below.
    merged[target_col_name] = merged[target_col_name].apply(
        lambda x: x if isinstance(x, list) else []
    )

    return merged


@task(log_prints=True, retries=1)
def ingest_work_logs(customer_code: str) -> pd.DataFrame:
    # ✅ Submit all 3 queries in parallel for faster execution
    towr_future = ingest_towr_data.submit(customer_code)
    csi_future = ingest_csi_survey_data.submit(customer_code)
    detail_future = ingest_detail_work_data.submit(customer_code)

    # Wait for all to complete
    df_towr = towr_future.result()
    df_csi = csi_future.result()
    df_detail_work = detail_future.result()

    date_columns_to_fix = [
        "작업일자",
        "설문기준일자",
        "서비스시작시간",
        "서비스종료시간",
    ]

    def sanitize_dates(df, cols_to_check):
        for col in cols_to_check:
            if col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(object).where(df[col].notnull(), None)
                    df[col] = df[col].apply(lambda x: str(x) if x is not None else None)
        return df

    # C. Service Detail
    df_detail_renamed = df_detail_work.rename(
        columns={
            "CONTTARG": "서비스내역_작업대상",
            "WORKCLAS": "서비스내역_작업구분",
            "WORKYN": "서비스내역_작업여부",
            "서비스작업번호": "JOIN_KEY_TOWR",
        }
    )

    # D. Survey
    df_survey_renamed = df_csi.rename(
        columns={
            "SDAY": "설문기준일자",
            "SURV_CD": "설문코드",
            "AVG_CSI_SCORE": "평균_CSI_점수",
            "CONTENTS": "설문내용",
            "서비스작업번호": "JOIN_KEY_TOWR",
        }
    )

    # E. Main TOWR (The Hub)
    df_towr_renamed = df_towr.rename(
        columns={
            "TOWRNO": "서비스작업번호",
            "CUSTTYPE": "고객유형",
            "WORKYMD": "작업일자",
            "WORKSEQN": "작업순번",
            "WORKTYPE": "작업유형",
            "CONFYN": "확정여부",
            "CANCYN": "취소여부",
            "VLYN": "유효여부",
            "CUSTCODE": "고객코드",
        }
    )

    df_towr_renamed = sanitize_dates(df_towr_renamed, date_columns_to_fix)
    df_survey_renamed = sanitize_dates(df_survey_renamed, date_columns_to_fix)
    # df_pswr_renamed = sanitize_dates(df_pswr_renamed, date_columns_to_fix)

    # ==========================================
    # 1. STITCHING (Reduced to 2 Steps)
    # ==========================================

    # --- STEP 1: Merge DETAIL into TOWR ---
    # Input:  df_towr_renamed (Raw Tower Data)
    # Output: df_towr_step1   (Tower + Details)
    df_towr_step1 = pack_children(
        parent_df=df_towr_renamed,
        child_df=df_detail_renamed,
        parent_key="서비스작업번호",
        child_key="JOIN_KEY_TOWR",
        target_col_name="서비스내역",
    )

    # --- STEP 2: Merge SURVEY into TOWR ---
    # Input:  df_towr_step1   (Result from Step 1)
    # Output: df_towr_final   (Tower + Details + Survey)
    df_towr_final = pack_children(
        parent_df=df_towr_step1,  # <--- Chain the previous step
        child_df=df_survey_renamed,
        parent_key="서비스작업번호",
        child_key="JOIN_KEY_TOWR",
        target_col_name="서비스_만족도",
    )

    # ==========================================
    # 2. GROUPING (Safe Method)
    # ==========================================
    if df_towr_final.empty:
        print("⚠️ No data found in df_towr_final. Returning empty result.")
        # Create an empty DataFrame with the CORRECT structure manually
        result_df = pd.DataFrame(columns=["고객코드", "작업이력"])

    else:
        # 2. We have data, so it is safe to group
        result_obj = df_towr_final.groupby("고객코드").apply(
            lambda x: x.drop(columns=["고객코드"], errors="ignore").to_dict(
                orient="records"
            )
        )

        # 3. Force Index Name (Safety)
        result_obj.index.name = "TEMP_INDEX_KEY"

        # 4. Reset Index
        result_df = result_obj.reset_index()

        result_df.columns = ["고객코드", "작업이력"]

    return result_df


@task(log_prints=True, retries=1)
def ingest_contracts(customer_code: str) -> pd.DataFrame:
    query = load_query_from_secret("contracts-query").format(
        custcode_list=customer_code,
        inference_date=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d"),
    )

    # Create fresh connection for each query to avoid "connection is closed" errors
    rodb_connection = get_rodb_connection()
    contracts_data = rodb_connection.execute_query(query)

    print(f"✅ Ingested {len(contracts_data)} contract records.")
    return contracts_data


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


@task(log_prints=True, retries=2)
def process_single_chunk(code_chunk: list, chunk_idx: int, today: datetime) -> dict:
    """
    Process a single chunk of customer codes.
    Returns a dict with the number of active customers processed and status.
    """
    chunk_start_time = time.time()
    code_formatted = ", ".join([f"'{code}'" for code in code_chunk])

    print(f"\n📦 Processing chunk {chunk_idx + 1} ({len(code_chunk)} codes)...")

    try:
        contract_db = ingest_contracts(code_formatted)
        customer_status_df = fetch_data(contract_db)

        active_customer_status_df = customer_status_df[
            customer_status_df["최종계약상태"] == 1
        ]

        # Skip if no active customers in this chunk
        if len(active_customer_status_df) == 0:
            print(f"⚠️ No active customers in chunk {chunk_idx + 1}, skipping...")
            return {"chunk_idx": chunk_idx, "active_count": 0, "status": "skipped"}

        code_formatted = ", ".join(
            [f"'{row['고객코드']}'" for _, row in active_customer_status_df.iterrows()]
        )

        # ✅ Parallel data fetching - both run at the same time
        work_db_future = ingest_work_logs.submit(code_formatted)
        purchase_db_future = ingest_purchase_logs.submit(code_formatted)

        # Wait for both to complete
        work_db = work_db_future.result()
        purchase_db = purchase_db_future.result()
        print(f"✅ Ingested {len(work_db)} work logs for chunk {chunk_idx + 1}.")

        chunk_df = active_customer_status_df.merge(work_db, on="고객코드", how="left")
        chunk_df = chunk_df.merge(purchase_db, on="고객코드", how="left")

        # ✅ Parallel ML inference - all 3 run at the same time
        load_future = process_and_load_chunk.submit(chunk_df, today)
        churn_future = predict_churn_for_chunk.submit(
            active_customer_status_df, work_db
        )
        recommend_future = recommend_users_for_chunk.submit(chunk_df)

        # Wait for all to complete
        load_future.result()
        chunk_churn_results = churn_future.result()
        chunk_recommendation_results = recommend_future.result()

        print(
            f"Churn Prediction Results for Chunk {chunk_idx + 1}:\n{chunk_churn_results}"
        )
        print(
            f"Recommendation Results for Chunk {chunk_idx + 1}:\n{chunk_recommendation_results}"
        )

        elapsed = time.time() - chunk_start_time
        print(f"✅ Chunk {chunk_idx + 1} completed in {elapsed:.2f}s")

        return {
            "chunk_idx": chunk_idx,
            "active_count": len(active_customer_status_df),
            "status": "success",
        }

    except Exception as e:
        print(f"❌ Error processing chunk {chunk_idx + 1}: {str(e)}")
        return {
            "chunk_idx": chunk_idx,
            "active_count": 0,
            "status": "error",
            "error": str(e),
        }


@flow(log_prints=True)
def analyze_current_user(max_workers: int = 20):
    today = datetime.now()
    print(
        f"📅 Starting data ingestion and inference for date: {today.strftime('%Y-%m')}"
    )

    # Preload all secrets at flow start to avoid parallel API calls
    print("🔑 Preloading query secrets...")
    for query_name in [
        "contracts-query",
        "csi-survey-query",
        "kodata-query",
        "pswr-detail-query",
        "pswr-query",
        "purchase-log-query",
        "towr-query",
        "user-ingestion-query",
        "work-detail-query",
    ]:
        load_query_from_secret(query_name)
    print("✅ All secrets loaded.")

    # prepare_database(today)
    customer_codes = ingest_users()
    print("📊 Total ingested customer codes:", len(customer_codes))
    print(f"Total unprocessed customer codes to analyze: {len(customer_codes)}")
    print(customer_codes[:5])

    # Get the list of customer codes
    code_list = customer_codes["고객코드"].tolist()
    total_codes = len(code_list)
    chunk_size = 2000  # Process 2000 customers per chunk

    print(f"Processing {total_codes} customer codes in chunks of {chunk_size}")
    print(f"Using {max_workers} parallel workers")

    # Create list of chunks
    chunks = list(chunk_list(code_list, chunk_size))
    total_chunks = len(chunks)
    print(f"Total chunks to process: {total_chunks}")

    # Submit all chunk processing tasks in parallel using Prefect's .submit()
    flow_start_time = time.time()
    futures = []
    results = []

    for chunk_idx, code_chunk in enumerate(chunks):
        future = process_single_chunk.submit(code_chunk, chunk_idx, today)
        futures.append((chunk_idx, future))

        # Small delay helps Prefect UI register tasks properly
        time.sleep(0.05)

        # Limit concurrent tasks
        if len(futures) >= max_workers:
            idx, completed_future = futures.pop(0)
            result = completed_future.result()
            results.append(result)
            print(f"📊 Chunk {result['chunk_idx'] + 1} result: {result['status']}")

    # Wait for remaining futures
    for idx, future in futures:
        result = future.result()
        results.append(result)
        print(f"📊 Chunk {result['chunk_idx'] + 1} result: {result['status']}")

    # Aggregate results
    total_active_customers = sum(r.get("active_count", 0) for r in results)
    successful_chunks = sum(1 for r in results if r.get("status") == "success")
    failed_chunks = sum(1 for r in results if r.get("status") == "error")

    elapsed_time = time.time() - flow_start_time
    print(f"\n{'=' * 50}")
    print("📊 Processing Summary:")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Successful: {successful_chunks}")
    print(f"   Failed: {failed_chunks}")
    print(f"   Total active customers: {total_active_customers}")
    print(f"   Total time: {elapsed_time:.2f}s")
    print(f"{'=' * 50}\n")

    recommender = UserRecommender()
    print("All chunks processed. Finalizing recommendations...")
    recommender.update_cluster_info(snapshot_month=today.strftime("%Y_%m"))


if __name__ == "__main__":
    analyze_current_user()
