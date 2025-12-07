from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from prefect import flow, task
from scipy.stats import entropy
from sqlalchemy import text

from src.core.db_utils import get_engine


@task(log_prints=True)
def get_confidence_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch confidence data from the database within the specified date range."""
    engine = get_engine()
    query = text("""
        SELECT model_confidence
        FROM analytics.voc_message_category
        WHERE created_at BETWEEN :start_date AND :end_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(
            query, conn, params={"start_date": start_date, "end_date": end_date}
        )
    return df


@task(log_prints=True)
def calculate_confidence_drift():
    print("Calculating confidence drift...")

    now = datetime.now()
    baseline_start = now - timedelta(days=64)
    baseline_end = now - timedelta(days=32)
    current_start = now - timedelta(days=32)

    baseline_scores = get_confidence_data(baseline_start, baseline_end)

    current_scores = get_confidence_data(current_start, now)

    if baseline_scores.empty or current_scores.empty:
        print("Warning: No data found for confidence drift calculation.")
        return 0.0

    bins = np.linspace(0, 1, 11)  # 10 bins between 0 and 1

    p_dist, _ = np.histogram(
        baseline_scores["model_confidence"], bins=bins, density=True
    )
    q_dist, _ = np.histogram(
        current_scores["model_confidence"], bins=bins, density=True
    )

    epsilon = 1e-10
    p_dist += epsilon
    q_dist += epsilon

    drift_scores = entropy(p_dist, q_dist)

    print(f"Calculated confidence drift: {drift_scores}")
    return drift_scores


@task(log_prints=True)
def get_update_records(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    print("Fetching records to update model health...")
    engine = get_engine()
    # get those where created_at != updated_at
    query = text("""
        SELECT *
        FROM analytics.voc_message_category
        WHERE created_at BETWEEN :start_date AND :end_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(
            query, conn, params={"start_date": start_date, "end_date": end_date}
        )
    print(f"Fetched {len(df)} records to update.")
    return df


@task(log_prints=True)
def get_model_indefinite_answers(
    start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    print("Fetching records with indefinite answers...")
    engine = get_engine()
    query = text("""
        SELECT *
        FROM analytics.voc_message_category
        WHERE main_category_1_name = '기타'
        AND created_at BETWEEN :start_date AND :end_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(
            query, conn, params={"start_date": start_date, "end_date": end_date}
        )
    print(f"Fetched {len(df)} indefinite answer records.")
    return df


@task(log_prints=True)
def calculate_indefinite_answer_rate():
    print("Calculating indefinite answer rate...")

    now = datetime.now()
    start_date = now - timedelta(days=30)

    total_records = get_update_records(start_date, now)
    indefinite_records = get_model_indefinite_answers(start_date, now)

    if total_records.empty:
        print("Warning: No records found for indefinite answer rate calculation.")
        return 0.0

    rate = len(indefinite_records) / len(total_records)
    print(f"Calculated indefinite answer rate: {rate:.4f}")
    return rate


@task(log_prints=True)
def calculate_update_rate():
    print("Calculating model update rate...")

    now = datetime.now()
    start_date = now - timedelta(days=30)

    total_records = get_update_records(start_date, now)
    updated_records = total_records[
        total_records["created_at"] != total_records["updated_at"]
    ]

    if total_records.empty:
        print("Warning: No records found for model update rate calculation.")
        return 0.0

    rate = len(updated_records) / len(total_records)
    print(f"Calculated model update rate: {rate:.4f}")
    return rate


@flow(log_prints=True)
def monitor_model_health():
    print("Monitoring model health...")
    drift = calculate_confidence_drift()
    print(f"Model Confidence Drift: {drift}")
    indefinite_rate = calculate_indefinite_answer_rate()
    print(f"Indefinite Answer Rate: {indefinite_rate}")
    update_rate = calculate_update_rate()
    print(f"Model Update Rate: {update_rate}")
    engine = get_engine()

    monitored_results = pd.DataFrame(
        {
            "model_name": ["cesco_sLLM_Qwen_3"],
            "model_version": ["1.0"],
            "unknown_rate": [indefinite_rate],
            "correction_rate": [update_rate],
            "confidence_drift": [drift],
        }
    )
    monitored_results.to_sql(
        "model_health", engine, if_exists="append", index=False, schema="monitoring"
    )


if __name__ == "__main__":
    monitor_model_health()
