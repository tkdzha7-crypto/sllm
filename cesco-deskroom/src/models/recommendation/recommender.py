import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from flows.common.db_utils import ensure_partition_exists, fast_bulk_insert, get_engine
from src.models.recommendation.category_mapper import (
    get_category_name,
    load_category_mapping,
)
from src.models.recommendation.mapping_constants import 업태_TO_분류, 종목_TO_분류


class UserRecommender:
    def __init__(
        self,
        inference_mode="current_user",
        model_path="src/models/simple_clustering_model_pca.pkl",
        mapping_dir="src/models",
    ):
        self.inference_mode = inference_mode
        self.model_path = model_path
        self.mapping_dir = mapping_dir
        self.업태_to_분류 = 업태_TO_분류
        self.종목_to_분류 = 종목_TO_분류

        # Load category mapping once at initialization
        self.category_mapping = load_category_mapping()

        # Model artifacts
        self.kmeans = None
        self.scaler = None
        self.pca = None
        self.le_sido = None
        self.le_sigungu = None
        self.le_대분류 = None
        self.le_중분류 = None
        self.le_소분류 = None
        self.le_세분류 = None
        self.le_업태 = None
        self.cluster_recommendations = None
        self.clustering_df = None

        self.feature_cols = [
            "위도",
            "경도",
            "시도_encoded",
            "시군구_encoded",
            "분류_PCA1",
            "분류_PCA2",
            "업태_encoded",
            "평균_면적_category",
        ]

        self.load_model_artifacts()

    def load_model_artifacts(self):
        """Load trained PCA model, encoders, PCA transformer, and cluster recommendations"""
        try:
            with open(self.model_path, "rb") as f:
                artifacts = pickle.load(f)

            self.kmeans = artifacts["kmeans"]
            self.scaler = artifacts["scaler"]
            self.pca = artifacts["pca"]
            self.le_sido = artifacts["le_sido"]
            self.le_sigungu = artifacts["le_sigungu"]
            self.le_대분류 = artifacts["le_대분류"]
            self.le_중분류 = artifacts["le_중분류"]
            self.le_소분류 = artifacts["le_소분류"]
            self.le_세분류 = artifacts["le_세분류"]
            self.le_업태 = artifacts["le_업태"]
            self.cluster_recommendations = artifacts["cluster_recommendations"]
            self.clustering_df = artifacts["clustering_df"]

            print("✅ PCA model loaded successfully")
            print("   - Features: 8 (with PCA dimensionality reduction)")
            print(f"   - Clusters: {self.kmeans.n_clusters}")

        except FileNotFoundError:
            print(
                "❌ Error: PCA model file not found. Please train the model first using the notebook."
            )
            sys.exit(1)

    def map_code_to_name(self, code):
        """Map industry code to name using cached mapping."""
        return get_category_name(code, self.category_mapping)

    def update_cluster_info(self, snapshot_month: str):
        # Ensure the user_monthly_features partition exists
        snapshot_date = datetime.strptime(snapshot_month, "%Y_%m").replace(day=1)
        ensure_partition_exists(
            target_date=snapshot_date,
            schema="source",
            table_name="user_monthly_features",
        )

        query = """
        SELECT
            ur.CCOD,
            ur.user_cluster,
            umf.user_information,
            umf.contract_info,
            umf.purchase_logs
        FROM analytics.user_recommendation_{snapshot_month} ur
            LEFT JOIN source.user_monthly_features_{snapshot_month} umf
            on ur.CCOD = umf.CCOD
        """
        query = query.format(snapshot_month=snapshot_month)
        engine = get_engine()
        df = pd.read_sql(query, engine)

        # If no data, skip cluster profile update
        if df.empty:
            print(
                f"⚠️ No data found for snapshot month {snapshot_month}. Skipping cluster profile update."
            )
            return

        # Pre-process JSON strings into objects
        def safe_json_loads(x):
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except json.JSONDecodeError:
                    return None
            return x

        df["user_information"] = df["user_information"].apply(safe_json_loads)
        df["contract_info"] = (
            df["contract_info"]
            .apply(safe_json_loads)
            .apply(lambda x: x if isinstance(x, list) else [])
        )
        df["purchase_logs"] = (
            df["purchase_logs"]
            .apply(safe_json_loads)
            .apply(lambda x: x if isinstance(x, list) else [])
        )

        cluster_summary = (
            df.groupby("user_cluster")
            .agg(
                cluster_size=("ccod", "nunique"),
                avg_contracts_num=(
                    "contract_info",
                    lambda x: np.mean([len(c) if c else 0 for c in x]),
                ),
                avg_purchase_num=(
                    "purchase_logs",
                    lambda x: np.mean([len(p) if p else 0 for p in x]),
                ),
                top_contracts=(
                    "contract_info",
                    lambda x: pd.Series(
                        [
                            item.get("계약대상")
                            for sublist in x.dropna()
                            for item in sublist
                            if item and "계약대상" in item
                        ]
                    )
                    .value_counts()
                    .head(5)
                    .index.tolist(),
                ),
                top_purchases=(
                    "purchase_logs",
                    lambda x: pd.Series(
                        [
                            item.get("service_name")
                            for sublist in x.dropna()
                            for item in sublist
                            if item and "service_name" in item
                        ]
                    )
                    .value_counts()
                    .head(5)
                    .index.tolist(),
                ),
                top_business_type=(
                    "user_information",
                    lambda x: pd.Series(
                        [
                            info.get("업태")
                            for info in x.dropna()
                            if info and "업태" in info
                        ]
                    )
                    .value_counts()
                    .head(5)
                    .index.tolist(),
                ),
                top_first_contract_code=(
                    "contract_info",
                    lambda x: pd.Series(
                        [
                            c[0].get("계약대상")
                            for c in x.dropna()
                            if c and len(c) > 0 and c[0] and "계약대상" in c[0]
                        ]
                    )
                    .value_counts()
                    .head(5)
                    .index.tolist(),
                ),
                contracts_distribution=(
                    "contract_info",
                    lambda x: pd.Series([len(c) for c in x.dropna() if c])
                    .value_counts()
                    .to_dict(),
                ),
                purchase_distribution=(
                    "purchase_logs",
                    lambda x: pd.Series([len(p) for p in x.dropna() if p])
                    .value_counts()
                    .to_dict(),
                ),
            )
            .reset_index()
        )
        print(cluster_summary.head())

        snapshot_month_date = datetime.now().strftime("%Y-%m-01")
        cluster_profile_df = pd.DataFrame()
        cluster_profile_df["snapshot_month"] = [snapshot_month_date] * len(
            cluster_summary
        )
        cluster_profile_df["cluster_id"] = cluster_summary["user_cluster"]
        cluster_profile_df["cluster_size"] = cluster_summary["cluster_size"]
        cluster_profile_df["avg_contracts_num"] = cluster_summary["avg_contracts_num"]
        cluster_profile_df["avg_purchase_num"] = cluster_summary["avg_purchase_num"]
        cluster_profile_df["top_contracts"] = cluster_summary["top_contracts"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
        cluster_profile_df["top_purchases"] = cluster_summary["top_purchases"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
        cluster_profile_df["top_business_type"] = cluster_summary[
            "top_business_type"
        ].apply(lambda x: json.dumps(x, ensure_ascii=False))
        cluster_profile_df["top_first_contract_code"] = cluster_summary[
            "top_first_contract_code"
        ].apply(lambda x: json.dumps(x, ensure_ascii=False))
        cluster_profile_df["contracts_distribution"] = cluster_summary[
            "contracts_distribution"
        ].apply(lambda x: json.dumps(x, ensure_ascii=False))
        cluster_profile_df["purchase_distribution"] = cluster_summary[
            "purchase_distribution"
        ].apply(lambda x: json.dumps(x, ensure_ascii=False))
        ensure_partition_exists(
            schema="analytics",
            table_name="cluster_profile",
            target_date=datetime.strptime(snapshot_month_date, "%Y-%m-%d"),
        )
        fast_bulk_insert(
            cluster_profile_df, table_name="cluster_profile", schema="analytics"
        )

    @staticmethod
    def extract_sido(address):
        """Extract 시도 from address"""
        if pd.isna(address):
            return None
        address = str(address).strip()

        sido_list = [
            "서울특별시",
            "서울",
            "부산광역시",
            "부산",
            "대구광역시",
            "대구",
            "인천광역시",
            "인천",
            "광주광역시",
            "광주",
            "대전광역시",
            "대전",
            "울산광역시",
            "울산",
            "세종특별자치시",
            "세종",
            "경기도",
            "경기",
            "강원도",
            "강원특별자치도",
            "강원",
            "충청북도",
            "충북",
            "충청남도",
            "충남",
            "전라북도",
            "전북",
            "전북특별자치도",
            "전라남도",
            "전남",
            "경상북도",
            "경북",
            "경상남도",
            "경남",
            "제주특별자치도",
            "제주",
        ]

        for sido in sido_list:
            if address.startswith(sido):
                if sido in ["서울", "서울특별시"]:
                    return "서울특별시"
                elif sido in ["부산", "부산광역시"]:
                    return "부산광역시"
                elif sido in ["대구", "대구광역시"]:
                    return "대구광역시"
                elif sido in ["인천", "인천광역시"]:
                    return "인천광역시"
                elif sido in ["광주", "광주광역시"]:
                    return "광주광역시"
                elif sido in ["대전", "대전광역시"]:
                    return "대전광역시"
                elif sido in ["울산", "울산광역시"]:
                    return "울산광역시"
                elif sido in ["세종", "세종특별자치시"]:
                    return "세종특별자치시"
                elif sido in ["경기", "경기도"]:
                    return "경기도"
                elif sido in ["강원", "강원도", "강원특별자치도"]:
                    return "강원특별자치도"
                elif sido in ["충북", "충청북도"]:
                    return "충청북도"
                elif sido in ["충남", "충청남도"]:
                    return "충청남도"
                elif sido in ["전북", "전라북도", "전북특별자치도"]:
                    return "전북특별자치도"
                elif sido in ["전남", "전라남도"]:
                    return "전라남도"
                elif sido in ["경북", "경상북도"]:
                    return "경상북도"
                elif sido in ["경남", "경상남도"]:
                    return "경상남도"
                elif sido in ["제주", "제주특별자치도"]:
                    return "제주특별자치도"
                return sido
        return None

    @staticmethod
    def extract_sigungu(address):
        """Extract 시군구 from address"""
        if pd.isna(address):
            return None
        address = str(address).strip()

        sido = UserRecommender.extract_sido(address)
        if sido:
            address = address.replace(sido, "").strip()

        parts = address.split()
        if len(parts) > 0:
            sigungu = parts[0]
            if "(" in sigungu:
                sigungu = sigungu.split("(")[0].strip()
            return sigungu
        return None

    @staticmethod
    def map_업종명_to_업태(업종명_value):
        """실제 데이터 분석 기반으로 개선된 업종명 → 업태 매핑"""
        if pd.isna(업종명_value):
            return None

        업종명 = str(업종명_value).strip()

        if any(
            kw in 업종명
            for kw in [
                "음식점",
                "한식",
                "중식",
                "일식",
                "양식",
                "분식",
                "치킨",
                "카페",
                "커피",
                "주점",
                "식당",
                "피자",
                "햄버거",
                "샌드위치",
                "베이커리",
            ]
        ):
            return "음식점업"

        if any(
            kw in 업종명
            for kw in [
                "소매",
                "슈퍼마켓",
                "편의점",
                "마트",
                "도매",
                "종합 소매",
                "빵류",
                "과자류",
                "육류",
                "문구",
                "화초",
                "식물",
                "주방용품",
                "가전",
                "의복",
                "상품",
            ]
        ):
            return "도소매"

        if any(kw in 업종명 for kw in ["제조", "생산", "공장", "제작", "인쇄", "제품"]):
            return "제조업"

        if any(kw in 업종명 for kw in ["교육", "학원", "훈련"]):
            return "교육업"

        if any(
            kw in 업종명
            for kw in [
                "의원",
                "병원",
                "한의원",
                "치과",
                "약국",
                "보건",
                "의료",
                "클리닉",
            ]
        ):
            return "보건업"

        if any(
            kw in 업종명 for kw in ["숙박", "호텔", "모텔", "펜션", "욕탕업", "찜질방"]
        ):
            return "숙박업"

        if any(
            kw in 업종명
            for kw in [
                "서비스",
                "미용",
                "세탁",
                "수리",
                "청소",
                "관리",
                "경비",
                "경호",
                "인테리어",
                "디자인",
                "복지",
                "사회",
            ]
        ):
            return "서비스업"

        if any(kw in 업종명 for kw in ["부동산", "임대", "중개", "대리", "빌딩"]):
            return "부동산업"

        if any(kw in 업종명 for kw in ["건설", "건축", "시공", "토목"]):
            return "건설업"

        if any(
            kw in 업종명
            for kw in [
                "운수",
                "운송",
                "배송",
                "택배",
                "물류",
                "화물",
                "자동차 운송",
                "창고",
            ]
        ):
            return "운수/물류업"

        if any(
            kw in 업종명 for kw in ["영화관", "극장", "공연", "오락", "체육", "레저"]
        ):
            return "문화/여가업"

        return "기타"

    @staticmethod
    def map_분류_to_업태(대분류, 중분류):
        """대분류와 중분류를 사용한 정확한 업태 매핑"""
        if pd.isna(대분류):
            return None

        대분류 = str(대분류).strip()
        중분류 = str(중분류).strip() if not pd.isna(중분류) else ""

        if 대분류 == "요식업체":
            return "음식점업"

        if 대분류 == "가정집":
            return "가정/주거"

        if 대분류 == "일반사업체":
            if any(kw in 중분류 for kw in ["공장", "제조"]):
                return "제조업"
            if any(kw in 중분류 for kw in ["교육", "학원", "훈련"]):
                return "교육업"
            if any(kw in 중분류 for kw in ["의료", "병원", "보건"]):
                return "보건업"
            if any(kw in 중분류 for kw in ["숙박", "호텔"]):
                return "숙박업"
            if any(kw in 중분류 for kw in ["판매", "유통", "마트", "상가"]):
                return "도소매"
            if any(kw in 중분류 for kw in ["빌딩", "부동산"]):
                return "부동산업"
            if any(kw in 중분류 for kw in ["복지", "사회"]):
                return "서비스업"
            if any(kw in 중분류 for kw in ["창고", "물류"]):
                return "운수/물류업"
            if "서비스" in 중분류:
                return "서비스업"
            return "기타사업체"

        return "기타"

    def map_업종명_to_분류(self, 업종명_value):
        """업종명 → 대/중/소/세분류 매핑"""

        if pd.isna(업종명_value):
            return {"대분류": None, "중분류": None, "소분류": None, "세분류": None}

        업종명 = str(업종명_value).strip()

        # 1. 업태 직접 매칭
        if 업종명 in self.업태_to_분류:
            return self.업태_to_분류[업종명].copy()

        # 2. 종목 직접 매칭
        if 업종명 in self.종목_to_분류:
            return self.종목_to_분류[업종명].copy()

        # 3. 업태 부분 매칭
        best_match = None
        best_length = 0
        for 업태, 분류 in self.업태_to_분류.items():
            if len(업태) >= 2:
                if 업태 in 업종명 and len(업태) > best_length:
                    best_match = 분류
                    best_length = len(업태)
                elif len(업종명) >= 4 and 업종명 in 업태 and len(업종명) > best_length:
                    best_match = 분류
                    best_length = len(업종명)

        if best_match is not None:
            return best_match.copy()

        # 4. 종목 부분 매칭
        best_match = None
        best_length = 0
        for 종목, 분류 in self.종목_to_분류.items():
            if 종목 and len(종목) >= 2:
                if 종목 in 업종명 and len(종목) > best_length:
                    best_match = 분류
                    best_length = len(종목)
                elif len(업종명) >= 4 and 업종명 in 종목 and len(업종명) > best_length:
                    best_match = 분류
                    best_length = len(업종명)

        if best_match is not None:
            return best_match.copy()

        return {"대분류": None, "중분류": None, "소분류": None, "세분류": None}

    @staticmethod
    def categorize_area(area):
        if not isinstance(area, (int, float)):
            return 0
        """Bin area into 6 categories"""
        if area <= 22:
            return 0
        elif area <= 50:
            return 1
        elif area <= 258:
            return 2
        elif area <= 1600:
            return 3
        elif area <= 4950:
            return 4
        else:
            return 5

    def recommend(self, customer_data):
        """Generate recommendations based on 8 features with PCA transformation"""

        # Step 1: Encode 대/중/소/세분류
        encoded_분류 = []
        for encoder, col_name in [
            (self.le_대분류, "대분류"),
            (self.le_중분류, "중분류"),
            (self.le_소분류, "소분류"),
            (self.le_세분류, "세분류"),
        ]:
            val = customer_data.get(col_name, "Unknown")
            if val in encoder.classes_:
                encoded_분류.append(encoder.transform([val])[0])
            else:
                encoded_분류.append(0)

        # Step 2: Apply PCA transformation to get 분류_PCA1 and 분류_PCA2
        분류_pca = self.pca.transform([encoded_분류])[0]

        # Step 3: Build feature vector with PCA components
        features = {}
        for col in self.feature_cols:
            if col == "평균_면적_category":
                area = customer_data.get("평균_면적", 50)
                features[col] = self.categorize_area(area)
            elif col == "시도_encoded":
                val = customer_data.get("시도명", "Unknown")
                features[col] = (
                    self.le_sido.transform([val])[0]
                    if val in self.le_sido.classes_
                    else 0
                )
            elif col == "시군구_encoded":
                val = customer_data.get("시군구명", "Unknown")
                features[col] = (
                    self.le_sigungu.transform([val])[0]
                    if val in self.le_sigungu.classes_
                    else 0
                )
            elif col == "분류_PCA1":
                features[col] = 분류_pca[0]
            elif col == "분류_PCA2":
                features[col] = 분류_pca[1]
            elif col == "업태_encoded":
                val = customer_data.get("업태", "Unknown")
                features[col] = (
                    self.le_업태.transform([val])[0]
                    if val in self.le_업태.classes_
                    else 0
                )
            elif col in customer_data.index:
                features[col] = customer_data[col]
            else:
                features[col] = 0

        X_new = np.array([features[col] for col in self.feature_cols]).reshape(1, -1)
        X_new_scaled = self.scaler.transform(X_new)
        cluster_id = self.kmeans.predict(X_new_scaled)[0]

        cluster_recs = self.cluster_recommendations[cluster_id]
        return {
            "cluster": cluster_id,
            "contract_recommendations": cluster_recs.get(
                "contract_recommendations", []
            ),
            "product_recommendations": cluster_recs.get("product_recommendations", []),
            "X_scaled": X_new_scaled,
            "pca_components": 분류_pca,
        }

    def run_inference(self, inference_df, output_dir="output_pca"):
        """Run inference on new customers using PCA model"""

        print("=" * 60)
        print("Simple Customer Segmentation - Inference (PCA Model)")
        print("=" * 60)

        # Load model artifacts
        print("\n📦 Loading PCA model artifacts...")
        self.load_model_artifacts()

        print(f"✅ Loaded {len(inference_df)} customers")

        # Extract location features
        print("\n🗺️  Extracting location features...")
        if "주소1" in inference_df.columns and "시도명" not in inference_df.columns:
            inference_df["시도명"] = inference_df["주소1"].apply(self.extract_sido)
        if "주소1" in inference_df.columns and "시군구명" not in inference_df.columns:
            inference_df["시군구명"] = inference_df["주소1"].apply(self.extract_sigungu)

        if (
            "업종명" not in inference_df.columns
            and "표준산업코드" in inference_df.columns
        ):
            inference_df["업종명"] = inference_df["표준산업코드"].apply(
                self.map_code_to_name
            )

        # Map 업종명 → 대/중/소/세분류 → 업태
        if "업종명" in inference_df.columns:
            print("🎯 업종명 → 대/중/소/세분류 매핑 중...")

            분류_results = inference_df["업종명"].apply(self.map_업종명_to_분류)
            inference_df["대분류"] = 분류_results.apply(lambda x: x["대분류"])
            inference_df["중분류"] = 분류_results.apply(lambda x: x["중분류"])
            inference_df["소분류"] = 분류_results.apply(lambda x: x["소분류"])
            inference_df["세분류"] = 분류_results.apply(lambda x: x["세분류"])

            inference_df["업태_from_분류"] = inference_df.apply(
                lambda row: self.map_분류_to_업태(row["대분류"], row["중분류"]), axis=1
            )
            inference_df["업태_from_키워드"] = inference_df["업종명"].apply(
                self.map_업종명_to_업태
            )
            inference_df["업태"] = inference_df["업태_from_분류"].fillna(
                inference_df["업태_from_키워드"]
            )

            mapped_count = inference_df["대분류"].notna().sum()
            print(f"✅ 업종명 → 분류 매핑: {mapped_count}/{len(inference_df)}")
            print(
                f"✅ 최종 업태 매핑: {inference_df['업태'].notna().sum()}/{len(inference_df)}"
            )

        elif "대분류" in inference_df.columns and "중분류" in inference_df.columns:
            print("🎯 Using existing classification data...")
            inference_df["업태"] = inference_df.apply(
                lambda row: self.map_분류_to_업태(row["대분류"], row["중분류"]), axis=1
            )
            print(
                f"✅ Classification-based: {inference_df['업태'].notna().sum()}/{len(inference_df)}"
            )

        elif "업종명" in inference_df.columns:
            print("⚠️  No mapping tables, using keyword-based mapping...")
            inference_df["업태"] = inference_df["업종명"].apply(self.map_업종명_to_업태)
            print(
                f"✅ Mapped 업태: {inference_df['업태'].notna().sum()}/{len(inference_df)}"
            )

        # Generate recommendations
        print("\n🔮 Generating recommendations with PCA model...")
        inference_output_rows = []

        # Calculate max distance for similarity score
        X_scaled_train = self.scaler.transform(
            self.clustering_df[self.feature_cols].values
        )
        max_dist = np.max(self.kmeans.transform(X_scaled_train))

        for i, (idx, customer) in enumerate(inference_df.iterrows()):
            if (i + 1) % 50 == 0:
                print(f"  Processing {i + 1}/{len(inference_df)}...")

            # Get first contract to exclude from recommendations
            first_contract_to_exclude = None
            cust_code = customer.get("고객코드", "")
            if cust_code:
                ground_truth_row = inference_df[inference_df["고객코드"] == cust_code]
                if not ground_truth_row.empty:
                    contracts_info_json = ground_truth_row.iloc[0].get(
                        "contracts_info", ""
                    )
                    if pd.notna(contracts_info_json) and contracts_info_json:
                        try:
                            contracts_list = json.loads(contracts_info_json)
                            contracts_with_dates = []
                            for c in contracts_list:
                                contract_target = c.get("계약대상", "")
                                contract_date = c.get("계약일자", "")
                                if contract_target and contract_date:
                                    try:
                                        date_obj = pd.to_datetime(contract_date)
                                        contracts_with_dates.append(
                                            {
                                                "target": contract_target,
                                                "date": date_obj,
                                            }
                                        )
                                    except Exception:
                                        contracts_with_dates.append(
                                            {
                                                "target": contract_target,
                                                "date": pd.Timestamp.max,
                                            }
                                        )
                            if contracts_with_dates:
                                contracts_with_dates.sort(key=lambda x: x["date"])
                                first_contract_to_exclude = contracts_with_dates[0][
                                    "target"
                                ]
                        except (json.JSONDecodeError, TypeError):
                            pass

            # Prepare customer data
            customer_data = pd.Series(
                {
                    "위도": customer.get("위도", 37.5)
                    if pd.notna(customer.get("위도"))
                    else 37.5,
                    "경도": customer.get("경도", 127.0)
                    if pd.notna(customer.get("경도"))
                    else 127.0,
                    "시도명": customer.get("시도명", "Unknown"),
                    "시군구명": customer.get("시군구명", "Unknown"),
                    "대분류": customer.get("대분류", "Unknown"),
                    "중분류": customer.get("중분류", "Unknown"),
                    "소분류": customer.get("소분류", "Unknown"),
                    "세분류": customer.get("세분류", "Unknown"),
                    "업태": customer.get("업태", "Unknown"),
                    "평균_면적": customer.get("건물규모", 50)
                    if pd.notna(customer.get("건물규모"))
                    else 50,
                }
            )

            # Get recommendations (with PCA transformation)
            recs = self.recommend(customer_data)

            # Calculate similarity
            cluster_id = recs["cluster"]
            X_new_scaled = recs["X_scaled"]
            distance = np.linalg.norm(
                X_new_scaled - self.kmeans.cluster_centers_[cluster_id]
            )
            similarity_score = 1 - (distance / max_dist)

            contract_recs_all = recs["contract_recommendations"][
                :10
            ]  # Get top 10 to re-sort
            product_recs_all = recs["product_recommendations"][:10]
            # Find most similar customer in cluster (closest by distance)
            cluster_data = self.clustering_df[
                self.clustering_df["cluster_pca"] == cluster_id
            ]
            cluster_size = len(cluster_data)

            contract_usage_data = []
            for contract in contract_recs_all:
                # Skip the first contract
                if contract == first_contract_to_exclude:
                    continue
                # Count unique customers who have this contract
                unique_customers = set()
                for _, row in cluster_data.iterrows():
                    if contract in row["계약코드_리스트"]:
                        unique_customers.add(row["고객코드"])
                usage_rate = (
                    (len(unique_customers) / cluster_size * 100)
                    if cluster_size > 0
                    else 0
                )
                contract_usage_data.append((contract, usage_rate))

            # Sort by usage rate (descending) and take top 3
            contract_usage_data.sort(key=lambda x: x[1], reverse=True)
            contract_recs = [c for c, _ in contract_usage_data[:3]]
            contract_usage_rates = [r for _, r in contract_usage_data[:3]]

            # Calculate purchase rates for ALL product recommendations (DEDUPLICATED)
            product_purchase_data = []
            for product in product_recs_all:
                # Count unique customers who purchased this product
                unique_customers = set()
                for _, row in cluster_data.iterrows():
                    if product in row["마이랩_상품명_리스트"]:
                        unique_customers.add(row["고객코드"])
                purchase_rate = (
                    (len(unique_customers) / cluster_size * 100)
                    if cluster_size > 0
                    else 0
                )
                product_purchase_data.append((product, purchase_rate))

            # Sort by purchase rate (descending) and take top 3
            product_purchase_data.sort(key=lambda x: x[1], reverse=True)
            product_recs = [p for p, _ in product_purchase_data[:3]]
            product_purchase_rates = [r for _, r in product_purchase_data[:3]]

            if len(cluster_data) > 0:
                # Calculate distance to all customers in cluster
                cluster_features = cluster_data[self.feature_cols].values
                cluster_scaled = self.scaler.transform(cluster_features)

                # Find closest customer
                distances = np.linalg.norm(cluster_scaled - X_new_scaled, axis=1)
                closest_idx = np.argmin(distances)
                similar_customer = cluster_data.iloc[closest_idx]
            else:
                similar_customer = None

            if self.inference_mode == "current_user":
                row = {
                    "CCOD": customer.get("고객코드", ""),
                    "snapshot_month": datetime.now().strftime("%Y-%m-01"),
                    "user_cluster": cluster_id,
                    "cluster_similarity": similarity_score,
                    "sim_CCOD": similar_customer["고객코드"]
                    if similar_customer is not None
                    else "",
                    "sim_user_name": similar_customer["고객명"]
                    if similar_customer is not None
                    else "",
                    "sim_user_contracts": json.dumps(
                        list(set(similar_customer["계약코드_리스트"])),
                        ensure_ascii=False,
                    )
                    if similar_customer is not None
                    and isinstance(similar_customer.get("계약코드_리스트"), list)
                    else "[]",  # JSONB
                    "sim_user_products": json.dumps(
                        list(set(similar_customer["마이랩_상품명_리스트"])),
                        ensure_ascii=False,
                    )
                    if similar_customer is not None
                    and isinstance(similar_customer.get("마이랩_상품명_리스트"), list)
                    else "[]",  # JSONB
                }
            else:
                row = {
                    "BZNO": customer.get("사업자번호", ""),
                    "ENP_NM": customer.get("상호명", ""),
                    "KEDCD": customer.get("KEDCD", ""),
                    "BZPL_CD": customer.get("BZPL_CD", ""),
                    "BZPL_SEQ": customer.get("BZPL_SEQ", ""),
                    "ENP_NP": customer.get("ENP_NP", ""),
                    "snapshot_month": datetime.now().strftime("%Y-%m-01"),
                    "user_cluster": cluster_id,
                    "cluster_similarity": similarity_score,
                    "sim_CCOD": similar_customer["고객코드"]
                    if similar_customer is not None
                    else "",
                    "sim_user_name": similar_customer["고객명"]
                    if similar_customer is not None
                    else "",
                    "sim_user_contracts": json.dumps(
                        list(set(similar_customer["계약코드_리스트"])),
                        ensure_ascii=False,
                    )
                    if similar_customer is not None
                    and isinstance(similar_customer.get("계약코드_리스트"), list)
                    else "[]",  # JSONB
                    "sim_user_products": json.dumps(
                        list(set(similar_customer["마이랩_상품명_리스트"])),
                        ensure_ascii=False,
                    )
                    if similar_customer is not None
                    and isinstance(similar_customer.get("마이랩_상품명_리스트"), list)
                    else "[]",  # JSONB
                }

            # Add recommendations with usage rate
            for j in range(3):
                if j < len(contract_recs):
                    contract = contract_recs[j]
                    usage_rate = contract_usage_rates[j]
                    row[f"rec_contract_{j+1}"] = contract
                    row[f"rec_contract_{j+1}_reason"] = (
                        f"고객 세그먼트 내 {usage_rate:.1f}%가 이용중"
                    )
                else:
                    row[f"rec_contract_{j+1}"] = ""
                    row[f"rec_contract_{j+1}_reason"] = ""

            for j in range(3):
                if j < len(product_recs):
                    product = product_recs[j]
                    purchase_rate = product_purchase_rates[j]
                    row[f"rec_product_{j+1}"] = product
                    row[f"rec_product_{j+1}_reason"] = (
                        f"고객 세그먼트 내 {purchase_rate:.1f}%가 구매함"
                    )
                else:
                    row[f"rec_product_{j+1}"] = ""
                    row[f"rec_product_{j+1}_reason"] = ""

            inference_output_rows.append(row)

        # Save results
        print("\n💾 Saving results...")
        os.makedirs(output_dir, exist_ok=True)
        inference_output_df = pd.DataFrame(inference_output_rows)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/output_potential_users_recommendations_pca_{timestamp}_known.csv"
        inference_output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        if self.inference_mode == "current_user":
            ensure_partition_exists(
                schema="analytics",
                table_name="user_recommendation",
                target_date=datetime.now(),
            )
            fast_bulk_insert(
                inference_output_df,
                table_name="user_recommendation",
                schema="analytics",
            )
        else:
            ensure_partition_exists(
                schema="analytics",
                table_name="potential_user_recommendation",
                target_date=datetime.now(),
            )
            fast_bulk_insert(
                inference_output_df,
                table_name="potential_user_recommendation",
                schema="analytics",
            )

        print(f"\n{'='*60}")
        print("✅ Inference complete!")
        print(f"{'='*60}")
        print("📊 Results:")
        print(f"   Total customers: {len(inference_output_df)}")
        print(f"   Clusters used: {inference_output_df['user_cluster'].nunique()}")
        print(f"   Output saved: {output_path}")
        print("\n📈 Cluster distribution:")
        print(inference_output_df["user_cluster"].value_counts().sort_index())
        print("\n💯 Similarity scores:")
        print(f"   Mean: {inference_output_df['cluster_similarity'].mean():.2%}")
        print(f"   Median: {inference_output_df['cluster_similarity'].median():.2%}")
        print(f"   Min: {inference_output_df['cluster_similarity'].min():.2%}")
        print(f"   Max: {inference_output_df['cluster_similarity'].max():.2%}")

        return inference_output_df


if __name__ == "__main__":
    recommender = UserRecommender()
    recommender.update_cluster_info(snapshot_month="2025_11")
    # recommender.run_inference(pd.read_csv('./user_contract_data.csv'), output_dir='.')
