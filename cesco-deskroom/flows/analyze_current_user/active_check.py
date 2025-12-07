import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from prefect import task

warnings.filterwarnings("ignore")
results_dir = "c:/Users/user/Documents/ml/cancellation-prediction/results"
print("✅ 라이브러리 임포트 완료")


@task(log_prints=True)
def fetch_data(customers_df):
    # 🔄 JSON 계약 정보를 DataFrame으로 변환

    print("🔄 계약 정보 처리 중...")

    # 계약 정보가 있는 고객만 필터링
    customers_with_contracts = customers_df[
        customers_df["contracts_info"].notna()
    ].copy()
    print(f"📊 계약 정보가 있는 고객: {len(customers_with_contracts):,}명")

    # JSON 데이터를 DataFrame으로 변환 (vectorized)
    def parse_contracts_safe(row):
        try:
            contracts = json.loads(row["contracts_info"])
            return [
                {"고객코드": row["고객코드"], "고객명": row["고객명"], **c}
                for c in contracts
            ]
        except (json.JSONDecodeError, TypeError):
            return []

    all_contracts = customers_with_contracts.apply(parse_contracts_safe, axis=1)
    contracts_records = [item for sublist in all_contracts for item in sublist]

    contracts_df = pd.DataFrame(contracts_records)
    # Ensure required columns exist
    required_cols = [
        "고객코드",
        "고객명",
        "계약일련번호",
        "계약일자",
        "시작일자",
        "종료일자",
        "해약일자",
        "해약여부",
        "해약일련번호",
        "면적",
        "계약대상",
    ]
    for col in required_cols:
        if col not in contracts_df.columns:
            contracts_df[col] = None
    print(f"✅ 총 {len(contracts_df):,}건의 계약 데이터 변환 완료")

    # 데이터 타입 변환 및 정리
    date_columns = ["계약일자", "시작일자", "종료일자", "해약일자"]
    for col in date_columns:
        contracts_df[col] = pd.to_datetime(contracts_df[col], errors="coerce")

    contracts_df.head()
    # 📋 해약여부관련_정리_v0_2.ipynb 로직 적용
    print("🔄 해약 로직 적용 중...")

    # Step 1: 데이터 정리 (유효하지 않은 고객코드 제외)
    df_1 = contracts_df[
        (contracts_df["고객코드"].notna())
        & (contracts_df["고객코드"] != "")
        & (contracts_df["고객코드"] != "AT7728")
    ].copy()

    print(f"📊 1단계: 유효한 고객 계약 데이터 {len(df_1):,}건")

    # Step 2: 날짜 컬럼 정리 및 변환
    today = datetime.now().strftime("%Y%m%d")

    # 해약일자 처리 - 해약일자가 없으면 해약여부가 FALSE인 경우 처리
    df_1["해약일자_processed"] = df_1["해약일자"].copy()

    # Step 3: 고객코드별, 계약대상별로 정렬하여 다음 계약 시작일 찾기
    df_2sort = df_1.sort_values(by=["고객코드", "계약대상", "시작일자"]).reset_index(
        drop=True
    )
    df_2sort["다음_계약시작일"] = df_2sort.groupby(["고객코드", "계약대상"])[
        "시작일자"
    ].shift(-1)

    print("📊 2단계: 다음 계약 시작일 매핑 완료")

    # Step 4: 해약일자 최종 결정 로직 (vectorized with np.select)
    cutoff_date = pd.to_datetime("2022-01-01")
    cond1 = (
        (df_2sort["해약일자_processed"].isna())
        & (df_2sort["종료일자"] >= cutoff_date)
        & (df_2sort["다음_계약시작일"].notna())
    )
    cond2 = (
        (df_2sort["해약일자_processed"].isna())
        & (df_2sort["종료일자"] >= cutoff_date)
        & (df_2sort["다음_계약시작일"].isna())
    )
    df_2sort["해약일자_final"] = np.select(
        [cond1, cond2],
        [df_2sort["다음_계약시작일"], df_2sort["시작일자"]],
        default=df_2sort["해약일자_processed"],
    )

    # Step 5: 오늘 날짜 이전의 종료일자인 경우 종료일자를 해약일자로 사용
    df_2sort["해약일자_final"] = np.where(
        (df_2sort["해약일자_final"].isna())
        & (df_2sort["종료일자"] < pd.to_datetime(today)),
        df_2sort["종료일자"],
        df_2sort["해약일자_final"],
    )

    print("📊 3단계: 해약일자 최종 결정 완료")

    # Step 6: 논리적 계약해지일 결정 (vectorized)
    today_dt = pd.to_datetime(today)
    cond_active = (df_2sort["해약일자_final"].isna()) & (
        df_2sort["종료일자"] > today_dt
    )
    df_2sort["논리_계약해지일"] = df_2sort["해약일자_final"].copy()
    df_2sort.loc[cond_active, "논리_계약해지일"] = today_dt

    # Step 7: 유지일수 계산
    df_2sort["유지일수"] = (
        pd.to_datetime(df_2sort["논리_계약해지일"]) - df_2sort["시작일자"]
    ).dt.days

    df_2sort[
        [
            "고객코드",
            "계약대상",
            "시작일자",
            "종료일자",
            "해약일자",
            "해약일자_final",
            "논리_계약해지일",
            "유지일수",
        ]
    ].head(10)

    # 📊 계약대상별 해약여부 집계 (해약여부관련_정리_v0_2.ipynb 로직)
    print("🔄 계약대상별 해약여부 집계 중...")

    # Step 8: 고객코드 + 계약대상별로 집계
    contract_target_summary = (
        df_2sort.groupby(["고객코드", "계약대상"])
        .agg(
            {
                "계약일련번호": ["count", "nunique"],
                "시작일자": "min",
                "종료일자": "max",
                "해약일자_final": ["count", "min", "max"],
                "논리_계약해지일": ["max", "min"],
                "유지일수": "sum",
            }
        )
        .reset_index()
    )

    # 컬럼명 정리
    contract_target_summary.columns = [
        "고객코드",
        "계약대상",
        "계약일련번호_cnt",
        "계약일련번호_unicnt",
        "최초계약_시작일자",
        "최근계약_종료일자",
        "해약일자_cnt",
        "최초_해약일자",
        "최근_해약일자",
        "최근_논리_계약해지일",
        "최초_논리_계약해지일",
        "유지일수_sum",
    ]

    # 계약대상별 해약여부 결정: 계약 건수와 해약 건수가 같으면 완전해약(1), 아니면 활성(0)
    contract_target_summary["계약대상별_해약여부"] = np.where(
        (
            contract_target_summary["계약일련번호_cnt"]
            - contract_target_summary["해약일자_cnt"]
        )
        > 0,
        0,
        1,
    )

    # Step 9: 고객별 최종 계약 상태 결정
    final_customer_status = (
        contract_target_summary.groupby("고객코드")
        .agg(
            {
                "계약대상": "count",  # 고객이 가진 계약대상 수
                "계약대상별_해약여부": "sum",  # 해약된 계약대상 수
            }
        )
        .reset_index()
    )

    final_customer_status.columns = ["고객코드", "총_계약대상수", "해약된_계약대상수"]

    # 최종 계약 상태: 활성 계약대상이 하나라도 있으면 활성(1), 모두 해약이면 해약(0)
    final_customer_status["최종계약상태"] = np.where(
        (
            final_customer_status["총_계약대상수"]
            - final_customer_status["해약된_계약대상수"]
        )
        > 0,
        1,
        0,
    )

    # 🔄 수정된 해약 로직 적용 (활성 계약 올바르게 식별)
    print("🔄 수정된 해약 로직 적용 중...")

    # 현재 날짜
    today_date = pd.to_datetime(today)
    print(f"📅 기준일: {today}")

    # 🔍 데이터 진단 추가
    print("\n🔍 데이터 진단:")
    print(f"   - 전체 계약 수: {len(df_2sort):,}건")
    print(f"   - 해약일자 null 개수: {df_2sort['해약일자'].isna().sum():,}건")
    print(
        f"   - 해약일자_final null 개수: {df_2sort['해약일자_final'].isna().sum():,}건"
    )
    print("   - 해약여부 값 분포:")
    print(df_2sort["해약여부"].value_counts())
    print(f"   - 종료일자 > 오늘: {(df_2sort['종료일자'] > today_date).sum():,}건")
    print("   - 종료일자 샘플:")
    print(df_2sort[["종료일자", "해약일자", "해약일자_final", "해약여부"]].head(10))

    # 수정된 로직: 계약이 활성인 조건
    # 1. 해약일자_final이 없고 (None 또는 NaT)
    # 2. 종료일자가 오늘 이후이거나 99991231(무기한) 이고
    # 3. 해약여부가 FALSE 인 경우

    # 해약여부 정규화 (다양한 형식 처리)
    df_2sort["해약여부_normalized"] = df_2sort["해약여부"].astype(str).str.upper()

    # 조건별로 카운트 확인
    cond1 = df_2sort["해약일자_final"].isna()
    cond2 = df_2sort["해약여부_normalized"].isin(["FALSE", "F", "N", "0"])
    cond3 = (df_2sort["종료일자"] > today_date) | (
        df_2sort["종료일자"].dt.strftime("%Y%m%d") == "99991231"
    )

    print("\n🔍 조건별 충족 현황:")
    print(f"   - 조건1 (해약일자_final 없음): {cond1.sum():,}건")
    print(f"   - 조건2 (해약여부 FALSE): {cond2.sum():,}건")
    print(f"   - 조건3 (종료일자 미래): {cond3.sum():,}건")
    print(f"   - 조건1 AND 조건2: {(cond1 & cond2).sum():,}건")
    print(f"   - 조건1 AND 조건3: {(cond1 & cond3).sum():,}건")
    print(f"   - 조건2 AND 조건3: {(cond2 & cond3).sum():,}건")

    df_2sort["is_active"] = cond1 | cond2 | cond3

    print("📊 수정된 로직에 따른 활성 계약 식별 완료")
    # 활성 계약 통계
    active_contracts_count = df_2sort["is_active"].sum()

    print(f"   - 활성 계약 수: {active_contracts_count:,}건")
    # 고객별 활성 계약 여부 집계
    customer_activity = (
        df_2sort.groupby("고객코드")
        .agg(
            {
                "is_active": [
                    "any",
                    #'sum'
                ],  # any: 활성 계약이 하나라도 있는지, sum: 활성 계약 수
                "계약대상": "nunique",  # 고객이 가진 총 계약대상 수
                "계약일련번호": "count",  # 고객이 가진 총 계약 수
            }
        )
        .reset_index()
    )

    # 컬럼명 정리
    customer_activity.columns = [
        "고객코드",
        "has_active_contract",
        #'active_contract_count',
        "unique_targets",
        "total_contracts",
    ]

    # 최종 고객 상태: 활성 계약이 하나라도 있으면 활성(1), 없으면 해약(0)
    customer_activity["최종계약상태"] = customer_activity["has_active_contract"].astype(
        int
    )
    # 원본 customers_df에 최종계약상태 정보 병합 (모든 컬럼 유지)
    customers_with_status = customers_df.merge(
        customer_activity[
            [
                "고객코드",
                "최종계약상태",
                # 'active_contract_count',
                "has_active_contract",
                "unique_targets",
                "total_contracts",
            ]
        ],
        on="고객코드",
        how="left",
    )

    # 병합되지 않은 고객들은 계약 정보가 없는 것으로 처리
    customers_with_status["최종계약상태"] = customers_with_status[
        "최종계약상태"
    ].fillna(0)
    # customers_with_status['active_contract_count'] = customers_with_status['active_contract_count'].fillna(0)
    customers_with_status["has_active_contract"] = customers_with_status[
        "has_active_contract"
    ].fillna(False)
    customers_with_status["unique_targets"] = customers_with_status[
        "unique_targets"
    ].fillna(0)
    # customers_with_status['total_contracts'] = customers_with_status['total_contracts'].fillna(0)

    # 전체 데이터 미리보기 (주요 컬럼들)
    display_cols = [
        "고객코드",
        "고객명",
        "유형대_명칭",
        "유형중_명칭",
        "주소1",
        "세분류",
        "소분류",
        "최종계약상태",
        # 'active_contract_count',
        "unique_targets",
    ]
    customers_with_status[display_cols].head()
    return customers_with_status
