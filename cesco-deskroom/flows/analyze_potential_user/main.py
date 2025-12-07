import pandas as pd
from prefect import flow, task
from prefect.blocks.system import Secret

from flows.common.db_utils import ensure_partition_exists, fast_bulk_insert
from src.dataloader import CescoRodbConnection
from src.models.recommendation.recommender import UserRecommender


def load_kodata_query() -> str:
    secret_block = Secret.load("kodata-query")
    return secret_block.get()


@task(log_prints=True, retries=1)
def ingest_new_users():
    print("Ingesting new potential users...")
    print("Connecting To RoDB...")
    rodb_connection = CescoRodbConnection()
    rodb_connection.connect()
    one_month_ago_kst = (pd.Timestamp.now() - pd.DateOffset(months=1)).strftime(
        "%Y-%m-%d"
    )
    print(f"Fetching users registered after {one_month_ago_kst}...")
    user_search_query = load_kodata_query().format(one_month_ago=one_month_ago_kst)
    print("Executing user search query...")
    new_users = rodb_connection.execute_query(user_search_query)
    print(f"Found {len(new_users)} new users.")
    if not new_users.empty:
        target_date = pd.Timestamp.now()
        ensure_partition_exists(
            target_date, schema="source", table_name="potential_user"
        )
        upload_df = pd.DataFrame(
            {
                "snapshot_month": target_date.strftime("%Y-%m-01"),
                "KEDCD": new_users["KEDCD"],
                "BZPL_CD": new_users["BZPL_CD"],
                "BZPL_SEQ": new_users["BZPL_SEQ"],
                "BZNO": new_users["사업자번호"],
                "ENP_NM": new_users["상호명"],
                "SIDO": new_users["시도명"],
                "SIGUNGU": new_users["시군구명"],
                "LAT": new_users["LAT"],
                "LOT": new_users["LOT"],
                "BZPL_NM": new_users["사업자명"],
                "BF_BZC_CD": new_users["표준산업코드"],
                "BSZE_METR": new_users["BSZE_METR"],
                "LSZE_METR": new_users["건물규모"],
                "RDNM_ADDR": new_users["RDNM_ADDR"],
            }
        )
        fast_bulk_insert(upload_df, table_name="potential_user", schema="source")
    else:
        print("No new users found to ingest.")

    return new_users


@task(log_prints=True)
def recommend_users_for_chunk(user_chunk: pd.DataFrame) -> pd.DataFrame:
    recommender = UserRecommender(inference_mode="potential_users")
    recommendation_results = recommender.run_inference(user_chunk)
    return recommendation_results


@flow(log_prints=True)
def analyze_potential_user():
    print("Analyzing potential users...")
    potential_users = ingest_new_users()
    print(f"Total potential users to process: {len(potential_users)}")
    ## chunk by 1000
    """
    if not potential_users.empty:
        chunk_size = 1000
        for i in range(0, len(potential_users), chunk_size):
            user_chunk = potential_users.iloc[i : i + chunk_size]
            results = recommend_users_for_chunk(user_chunk)
            print(f"Processed chunk {i // chunk_size + 1}: {len(user_chunk)} users")

    else:
        print("No potential users to process for recommendations.")
    """


if __name__ == "__main__":
    analyze_potential_user()
