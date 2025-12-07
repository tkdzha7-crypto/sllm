import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from prefect import flow, task

from flows.analyze_current_user.active_check import fetch_data
from flows.analyze_current_user.queries2 import (
    CSI_SURVEY_QUERY,
    DETAIL_WORK_QUERY,
    PSWR_DETAIL_QUERY,
    PSWR_QUERY,
    TOWR_QUERY,
)
from flows.common.db_utils import ensure_partition_exists, fast_bulk_insert, get_engine
from flows.common.queries import (
    CONTRACTS_QUERY_TEMPLATE,
    PURCHASE_LOGS_QUERY_TEMPLATE,
    USER_INGESTION_QUERY,
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


def get_processed_customers():
    db_engine = get_engine()
    query = "SELECT DISTINCT ccod FROM source.user_monthly_features"
    processed_customers = pd.read_sql(query, db_engine)
    return set(processed_customers["ccod"].tolist())


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
def ingest_towr_data(customer_code: str) -> pd.DataFrame:
    query = TOWR_QUERY.format(
        code_list=customer_code,
        inference_date=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d"),
    )

    bidb_connection = get_bidb_connection()
    towr_data = bidb_connection.execute_query(query)

    print(f"✅ Ingested {len(towr_data)} TOWR records.")
    return towr_data


@task(log_prints=True, retries=1)
def ingest_csi_survey_data(customer_code: str) -> pd.DataFrame:
    query = CSI_SURVEY_QUERY.format(
        code_list=customer_code,
        inference_date=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d"),
    )

    bidb_connection = get_bidb_connection()
    csi_survey_data = bidb_connection.execute_query(query)

    print(f"✅ Ingested {len(csi_survey_data)} CSI Survey records.")
    return csi_survey_data


@task(log_prints=True, retries=1)
def ingest_detail_work_data(customer_code: str) -> pd.DataFrame:
    query = DETAIL_WORK_QUERY.format(
        code_list=customer_code,
        inference_date=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d"),
    )

    bidb_connection = get_bidb_connection()
    detail_work_data = bidb_connection.execute_query(query)

    print(f"✅ Ingested {len(detail_work_data)} Detail Work records.")
    return detail_work_data


@task(log_prints=True, retries=1)
def ingest_pswr_data(customer_code: str) -> pd.DataFrame:
    query = PSWR_QUERY.format(
        code_list=customer_code,
        inference_date=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d"),
    )

    bidb_connection = get_bidb_connection()
    pwsr_data = bidb_connection.execute_query(query)

    print(f"✅ Ingested {len(pwsr_data)} PWSR records.")
    return pwsr_data


@task(log_prints=True, retries=1)
def ingest_pswr_detail_data(customer_code: str) -> pd.DataFrame:
    query = PSWR_DETAIL_QUERY.format(
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
    df_towr = ingest_towr_data(customer_code)
    df_csi = ingest_csi_survey_data(customer_code)
    df_detail_work = ingest_detail_work_data(customer_code)
    # df_pswr = ingest_pswr_data(customer_code)
    # df_pswr_detail = ingest_pswr_detail_data(customer_code)

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

    """
    df_val_renamed = df_pswr_detail.rename(columns={
        'DIVSENUM': '구획번호',
        'PESTCD': '해충코드',
        'PSOCTPCD': '상태코드',
        'OCPESCNT': '개체수',
        'DOCCNT': '해충마리수',
        '개인작업번호': 'JOIN_KEY_PSWR' # Keep key for joining, drop later
    })

    # B. PSWR Header
    df_pswr_renamed = df_pswr.rename(columns={
        'PSWRNO': '개인작업번호',
        'FRTIME': '서비스시작시간',
        'TOTIME': '서비스종료시간',
        'CONFYN': '확정여부',
        '서비스작업번호': 'JOIN_KEY_TOWR'
    })
    """
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
    """
    df_pswr_full = pack_children(
    parent_df=df_pswr_renamed,
    child_df=df_val_renamed,
    parent_key='개인작업번호',
    child_key='JOIN_KEY_PSWR',
    target_col_name='개인작업상세')
    """
    """
    df_towr_step1 = pack_children(
    parent_df=df_towr_renamed,
    child_df=df_pswr_full,
    parent_key='서비스작업번호',
    child_key='JOIN_KEY_TOWR',
    target_col_name='작업내역'
)
    """

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

        # 5. Rename Columns
        # This line crashed before because an empty groupby kept 12 columns.
        # Now it only runs if we actually grouped successfully.
        result_df.columns = ["고객코드", "작업이력"]

    # result_df['작업이력'] = result_df['작업이력_'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    """
    df_pswr_detail_grouped = df_pswr_detail.groupby('개인작업번호').apply(
        lambda x: x.to_dict(orient='records')
    ).reset_index(name = '개인작업상세')

    df_pswr = df_pswr.merge(
        df_pswr_detail_grouped,
        on = '개인작업번호',
        how = 'left'
    )
    df_pswr_grouped = df_pswr.groupby('서비스작업번호').apply(
        lambda x: x.to_dict(orient='records')
    ).reset_index(name = '작업내역')

    df_detail_work_grouped = df_detail_work.groupby('서비스작업번호').apply(
        lambda x: x.to_dict(orient='records')
    ).reset_index(name = '서비스내역')

    df_survey_grouped = df_csi.groupby('서비스작업번호').apply(
        lambda x: x.to_dict(orient='records')
    ).reset_index(name = '서비스_만족도')
    df_master = df_towr.merge(
        df_pswr_grouped, on = '서비스작업번호', how='left')

    df_master = df_master.merge(
        df_detail_work_grouped, on = '서비스작업번호', how='left')

    df_master = df_master.merge(
        df_survey_grouped, on = '서비스작업번호', how='left')

    result_df = df_master.groupby('고객코드').apply(
        lambda x: x.to_dict(orient='records'))
    """
    return result_df


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
    print("📊 Total ingested customer codes:", len(customer_codes))
    processed_customers = get_processed_customers()
    customer_codes = customer_codes[
        ~customer_codes["고객코드"].isin(processed_customers)
    ]
    print(f"Total unprocessed customer codes to analyze: {len(customer_codes)}")
    print(customer_codes[:5])

    # Get the list of customer codes
    code_list = customer_codes["고객코드"].tolist()
    # code_list = ['H25K007078', 'YU4623']
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

            # Skip if no active customers in this chunk
            if len(active_customer_status_df) == 0:
                print(f"⚠️ No active customers in chunk {chunk_idx + 1}, skipping...")
                continue

            code_formatted = ", ".join(
                [
                    f"'{row['고객코드']}'"
                    for _, row in active_customer_status_df.iterrows()
                ]
            )

            work_db = ingest_work_logs(code_formatted)
            ## save work_db for debug
            work_db.to_csv(f"work_db_chunk_{chunk_idx + 1}.csv", index=False)
            print(
                f"✅ Ingested work logs for active customers in chunk {chunk_idx + 1}."
            )
            print(len(work_db))
            purchase_db = ingest_purchase_logs(code_formatted)
            num_active_customers += len(active_customer_status_df)
            print(f"Current aggregated active customers: {num_active_customers}")
            chunk_df = active_customer_status_df.merge(
                work_db, on="고객코드", how="left"
            )
            chunk_df = chunk_df.merge(purchase_db, on="고객코드", how="left")
            process_and_load_chunk(chunk_df, today)
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
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ Error processing chunk {chunk_idx + 1}: {str(e)}")
            break
    recommender = UserRecommender()
    print("All chunks processed. Finalizing recommendations...")
    recommender.update_cluster_info(snapshot_month=today.strftime("%Y_%m"))


if __name__ == "__main__":
    analyze_current_user()
