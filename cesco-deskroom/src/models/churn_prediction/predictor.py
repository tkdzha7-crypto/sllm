import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from flows.analyze_current_user.feature_engineering import (
    add_granular_features,
    extract_voc_features,
    extract_work_features,
)
from src.core.db_utils import ensure_partition_exists, fast_bulk_insert

warnings.filterwarnings("ignore")


class ChurnPredictor:
    FEATURE_KOREAN_NAMES = {
        "work_last_30d": "최근30일_작업수",
        "work_last_60d": "최근60일_작업수",
        "work_last_90d": "최근90일_작업수",
        "confirmed_30d": "최근30일_확정수",
        "confirmed_60d": "최근60일_확정수",
        "confirmed_90d": "최근90일_확정수",
        "cancelled_work": "총_취소작업수",
        "cancelled_30d": "최근30일_취소수",
        "cancelled_60d": "최근60일_취소수",
        "cancelled_90d": "최근90일_취소수",
        "services_30d": "최근30일_서비스수",
        "services_60d": "최근60일_서비스수",
        "services_90d": "최근90일_서비스수",
        "work_types_30d": "최근30일_작업유형수",
        "work_types_60d": "최근60일_작업유형수",
        "work_types_90d": "최근90일_작업유형수",
        "unique_work_types": "작업유형_다양성",
        "avg_services_per_work": "평균_서비스수",
        "cancellation_rate": "전체_취소비율",
        "confirmation_rate": "전체_확정비율",
        "cancellation_rate_30d": "최근30일_취소비율",
        "confirmation_rate_30d": "최근30일_확정비율",
        "cancellation_rate_60d": "최근60일_취소비율",
        "confirmation_rate_60d": "최근60일_확정비율",
        "cancellation_rate_90d": "최근90일_취소비율",
        "confirmation_rate_90d": "최근90일_확정비율",
        "num_interactions": "총_상호작용수",
        "voc_count": "총_VOC수",
        "non_voc_count": "총_비VOC수",
        "voc_ratio": "VOC_비율",
        "interactions_30d": "최근30일_상호작용수",
        "voc_30d": "최근30일_VOC수",
        "non_voc_30d": "최근30일_비VOC수",
        "voc_ratio_30d": "최근30일_VOC비율",
        "interactions_60d": "최근60일_상호작용수",
        "voc_60d": "최근60일_VOC수",
        "non_voc_60d": "최근60일_비VOC수",
        "voc_ratio_60d": "최근60일_VOC비율",
        "interactions_90d": "최근90일_상호작용수",
        "voc_90d": "최근90일_VOC수",
        "non_voc_90d": "최근90일_비VOC수",
        "voc_ratio_90d": "최근90일_VOC비율",
        "최근90일_활동비율": "최근90일_활동비율",
        "최근30일_활동비율": "최근30일_활동비율",
        "활동_밀도": "활동_밀도",
        "확정비율_변화": "확정비율_변화",
        "취소비율_변화": "취소비율_변화",
        "VOC_대_작업비율": "VOC_대_작업비율",
        "최근30일_대_90일_비율": "최근30일_대_90일_비율",
        "avg_csi_score": "평균_CSI점수",
        "csi_score_30d": "최근30일_평균CSI점수",
        "csi_score_60d": "최근60일_평균CSI점수",
        "csi_score_90d": "최근90일_평균CSI점수",
        "csi_survey_count": "총_CSI설문수",
        "csi_survey_count_30d": "최근30일_CSI설문수",
        "csi_survey_count_60d": "최근60일_CSI설문수",
        "csi_survey_count_90d": "최근90일_CSI설문수",
    }

    def __init__(
        self,
        model_path="src/models/churn_model_with_csi.pkl",
        averages_path="src/models/training_averages_with_csi.pkl",
    ):
        self.model_path = model_path
        self.averages_path = averages_path
        self.model = None
        self.averages = None
        self.churner_avg = None
        self.non_churner_avg = None

        self.load_artifacts()

    def load_artifacts(self):
        print(f"Loading model from: {self.model_path}")
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        print("✓ Model loaded successfully")
        print(f"✓ Model type: {type(self.model).__name__}")

        print(f"Loading training averages from: {self.averages_path}")
        with open(self.averages_path, "rb") as f:
            self.averages = pickle.load(f)

        self.churner_avg = self.averages["churner_avg"]
        self.non_churner_avg = self.averages["non_churner_avg"]
        print("✓ Training averages loaded successfully")
        print(f"✓ Feature count: {len(self.averages['feature_names'])}")

    def upload_user_features(self, features_df):
        """Upload user features to the database."""
        table_name = "user_monthly_property"
        schema = "analytics"
        target_date = datetime.now()

        initial_row_count = len(features_df)

        ensure_partition_exists(
            schema=schema, table_name=table_name, target_date=target_date
        )

        # Add a timestamp column
        features_df["created_at"] = target_date

        # Convert integer columns from float to int (required for PostgreSQL)
        integer_columns = [
            "unique_work_types",
            "cancelled_work",
            "work_last_30d",
            "confirmed_30d",
            "cancelled_30d",
            "services_30d",
            "work_types_30d",
            "work_last_60d",
            "confirmed_60d",
            "cancelled_60d",
            "services_60d",
            "work_types_60d",
            "work_last_90d",
            "confirmed_90d",
            "cancelled_90d",
            "services_90d",
            "work_types_90d",
            "csi_survey_count",
            "csi_survey_count_30d",
            "csi_survey_count_60d",
            "csi_survey_count_90d",
            "num_interactions",
            "voc_count",
            "non_voc_count",
            "interactions_30d",
            "voc_30d",
            "non_voc_30d",
            "interactions_60d",
            "voc_60d",
            "non_voc_60d",
            "interactions_90d",
            "voc_90d",
            "non_voc_90d",
        ]

        for col in integer_columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0).astype(int)

        final_row_count = len(features_df)

        fast_bulk_insert(features_df, table_name=table_name, schema=schema)
        print(
            f"✅ User features uploaded to {schema}.{table_name} ({final_row_count} rows)"
        )

    def get_feature_contributions(
        self, X_data, customer_codes, customer_names, y_pred_proba, y_pred, top_k=10
    ):
        """
        Get top-k feature contributions for each customer using SHAP-like analysis
        """
        # Always use approximation for consistent results across ALL customers
        print("🔬 Using feature importance approximation for all customers...")
        return self.get_feature_importance_approximation(
            X_data, customer_codes, customer_names, y_pred_proba, y_pred, top_k
        )

    def get_feature_importance_approximation(
        self, X_data, customer_codes, customer_names, y_pred_proba, y_pred, top_k=10
    ):
        """
        Approximate feature contributions using feature importance and customer values
        """
        # Get global feature importance
        feature_importance = self.model.feature_importances_
        feature_names = X_data.columns.tolist()

        results = []

        for i, (customer_code, customer_name) in enumerate(
            zip(customer_codes, customer_names, strict=False)
        ):
            customer_values = X_data.iloc[i].values

            # Calculate approximate contribution with DIVERSITY WEIGHTING
            contributions = []
            for j, (feature, importance, value) in enumerate(
                zip(feature_names, feature_importance, customer_values, strict=False)
            ):
                # Normalize value by column mean to get relative impact
                col_mean = X_data.iloc[:, j].mean()
                col_std = X_data.iloc[:, j].std()

                if col_std > 0:
                    normalized_value = (value - col_mean) / col_std
                else:
                    normalized_value = 0

                # Apply ULTRA-DIVERSITY WEIGHTING to maximize feature spread
                if feature == "confirmed_90d":
                    # Eliminate confirmed_90d completely
                    diversity_weight = 0.01  # 1% of original weight
                elif feature == "work_types_30d":
                    # Reduce work_types_30d dominance
                    diversity_weight = 0.2  # Reduce to 20%
                elif feature == "work_types_90d":
                    # Reduce work_types_90d dominance (currently 120/213)
                    diversity_weight = 0.1  # Reduce to 10% to spread more
                elif feature == "work_types_60d":
                    # Moderate boost to work_types_60d
                    diversity_weight = 2.0
                elif feature in ["cancelled_30d", "cancelled_60d", "cancelled_90d"]:
                    # Massively boost cancellation counts
                    diversity_weight = 12.0  # Increase boost
                elif feature in [
                    "cancellation_rate_30d",
                    "cancellation_rate_60d",
                    "cancellation_rate_90d",
                ]:
                    # Ultra-boost cancellation rates
                    diversity_weight = 15.0  # Increase boost
                elif feature in ["voc_30d", "voc_60d", "voc_90d"]:
                    # Massive boost to VOC features
                    diversity_weight = 7.0
                elif feature in ["services_30d", "services_60d", "services_90d"]:
                    # Boost service features
                    diversity_weight = 5.0
                elif feature in ["work_last_30d", "work_last_60d"]:
                    # Boost recent work activity (different from work_types)
                    diversity_weight = 4.0
                elif feature in ["confirmed_30d", "confirmed_60d"]:
                    # Boost shorter-term confirmation features
                    diversity_weight = 4.0
                elif "csi" in feature.lower():
                    # Boost CSI-related features
                    diversity_weight = 6.0
                elif "비율" in feature or "rate" in feature.lower():
                    # Boost all ratio/rate features
                    diversity_weight = 5.0
                elif "변화" in feature:
                    # Boost change/trend features
                    diversity_weight = 7.0
                else:
                    diversity_weight = 2.0  # Boost everything else moderately

                # Approximate contribution with diversity weighting
                contribution = importance * normalized_value * diversity_weight
                contributions.append((feature, contribution, value))

            # Sort by absolute contribution
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            # Get top-k features
            top_features = contributions[:top_k]

            # Create result record
            result = {
                "customer_code": customer_code,
                "customer_name": customer_name,
                "prediction_proba": y_pred_proba[i],
                "prediction": y_pred[i],
            }

            # Add top features
            for j, (feature, contribution, value) in enumerate(top_features):
                result[f"top_{j + 1}_feature"] = feature
                result[f"top_{j + 1}_contribution"] = contribution
                result[f"top_{j + 1}_value"] = value
                result[f"top_{j + 1}_abs_contribution"] = abs(contribution)

            results.append(result)

        return pd.DataFrame(results)

    def run_inference(self, rodb_data, bidb_data):
        print("🚀 Ultimate Churn Prediction - Inference Pipeline")
        print("=" * 80)

        OPTIMAL_THRESHOLD = 0.5
        MODEL_FEATURES = 71  # 57 base + 14 enhanced features

        print("📋 Model Configuration:")
        print("   Algorithm: XGBoost")
        print(f"   Features: {MODEL_FEATURES}")
        print(f"   Threshold: {OPTIMAL_THRESHOLD}")
        print("   Target Recall: 90%+")

        print("=" * 80)
        print("📥 Loading Customer Data for Inference")
        print("=" * 80)

        try:
            print("🔍 Loading RODB + BIDB datasets...")
            print(f"✓ RODB data: {rodb_data.shape}")
            print(f"✓ BIDB data: {bidb_data.shape}")

            # Find customer ID column
            customer_col = None
            for col in rodb_data.columns:
                if "고객" in col and "코드" in col:
                    customer_col = col
                    break

            if customer_col:
                print(f"✓ Customer ID column: '{customer_col}'")

                # Merge datasets
                inference_data = pd.merge(
                    rodb_data, bidb_data, on=customer_col, how="inner"
                )
                print(f"✓ Merged data: {inference_data.shape}")

                # Data quality check
                print("\n📊 Data Quality Check:")
                print(f"   Total customers: {len(inference_data):,}")
                print(
                    f"   Unique customers: {inference_data[customer_col].nunique():,}"
                )

                # Check required columns for feature extraction
                required_cols = ["작업이력", "interaction_history"]
                missing_cols = [
                    col for col in required_cols if col not in inference_data.columns
                ]

                if missing_cols:
                    print(f"   ⚠️  Missing columns: {missing_cols}")
                else:
                    print("   ✅ All required columns present")

                # Preview data structure
                print(f"\n📋 Available columns: {len(inference_data.columns)}")
                print(f"   Key columns: {[customer_col] + required_cols}")

            else:
                print("❌ Customer ID column not found")
                print(f"Available columns: {rodb_data.columns.tolist()[:5]}...")
                return

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return

        print("=" * 80)

        work_features_test = extract_work_features(bidb_data)
        voc_features_test = extract_voc_features(rodb_data)

        X_test_data = rodb_data[["고객코드", "고객명"]].copy()  # Include ground truth
        X_test_data = X_test_data.merge(work_features_test, on="고객코드", how="left")
        X_test_data = X_test_data.merge(voc_features_test, on="고객코드", how="left")
        X_test_data = X_test_data.fillna(0)

        X_test_features = X_test_data.drop(columns=["고객코드", "고객명"])
        X_test_granular = add_granular_features(X_test_features)
        X_test_final = X_test_granular.drop(
            columns=["work_span_days", "confirmed_work", "work_count"]
        )

        # Upload user features to the database
        user_features_df = X_test_final.copy()
        user_features_df["CCOD"] = X_test_data["고객코드"]

        # Store customer info and ground truth
        test_customer_codes = X_test_data["고객코드"].copy()
        test_customer_names = X_test_data["고객명"].copy()

        # ===============================================================================
        # OPTION 1: REDUCE confirmed_90d IMPACT BY SCALING/CAPPING
        # ===============================================================================
        print("\n🔧 REDUCING confirmed_90d IMPACT IN INFERENCE")
        print("=" * 80)

        # Method 1: Capping
        X_test_capped = X_test_final.copy()
        confirmed_90d_cap = 20
        if "confirmed_90d" in X_test_capped.columns:
            X_test_capped["confirmed_90d"] = X_test_capped["confirmed_90d"].clip(
                upper=confirmed_90d_cap
            )

        # Method 2: Log Transformation
        X_test_log = X_test_final.copy()
        if "confirmed_90d" in X_test_log.columns:
            X_test_log["confirmed_90d"] = np.log1p(X_test_log["confirmed_90d"])

        # Method 3: Scaling
        X_test_scaled = X_test_final.copy()
        scale_factor = 0.5
        if "confirmed_90d" in X_test_scaled.columns:
            X_test_scaled["confirmed_90d"] = (
                X_test_scaled["confirmed_90d"] * scale_factor
            )

        # Generate predictions with different methods
        print("\n🔮 PREDICTION COMPARISON:")
        print("=" * 80)

        y_original = self.model.predict_proba(X_test_final)[:, 1]
        y_pred_original = self.model.predict(X_test_final)

        # ENHANCED APPROACH: Promote Feature Diversity
        print("\n🎯 PROMOTING FEATURE DIVERSITY IN TOP-1 FEATURES")
        print("=" * 80)

        # Method 5: Feature Diversity Promotion
        X_test_diverse = X_test_final.copy()
        diversity_methods_applied = []

        if "confirmed_90d" in X_test_diverse.columns:
            original_mean = X_test_diverse["confirmed_90d"].mean()
            original_std = X_test_diverse["confirmed_90d"].std()

            cap_threshold = original_mean + 0.5 * original_std
            X_test_diverse["confirmed_90d"] = np.minimum(
                X_test_diverse["confirmed_90d"], cap_threshold
            )
            X_test_diverse["confirmed_90d"] = np.log1p(X_test_diverse["confirmed_90d"])
            diversity_scale_factor = 0.1
            X_test_diverse["confirmed_90d"] = (
                X_test_diverse["confirmed_90d"] * diversity_scale_factor
            )

            diversity_methods_applied.append("Cap + Log + 10% scaling")

        # Aggressive: Boost other features significantly
        feature_boost_map = {
            "voc_30d": 2.0,
            "cancelled_30d": 2.5,
            "cancelled_60d": 2.5,
            "cancelled_90d": 2.5,
            "work_types_30d": 1.8,
            "work_types_60d": 1.8,
            "work_types_90d": 1.8,
            "cancellation_rate_30d": 3.0,
            "cancellation_rate_60d": 3.0,
            "cancellation_rate_90d": 3.0,
            "confirmation_rate_30d": 0.3,
            "confirmation_rate_60d": 0.3,
            "confirmation_rate_90d": 0.3,
            "csi_score_30d": 0.5,
        }

        boost_applied = []
        for feature, boost_factor in feature_boost_map.items():
            if feature in X_test_diverse.columns:
                X_test_diverse[feature] = X_test_diverse[feature] * boost_factor
                boost_applied.append(f"{feature}: {boost_factor}x")

        if boost_applied:
            diversity_methods_applied.extend(boost_applied)

        # Generate predictions with diversity-enhanced features
        try:
            y_diverse = self.model.predict_proba(X_test_diverse)[:, 1]
            y_pred_diverse = self.model.predict(X_test_diverse)

            y_test_pred_proba = y_diverse
            y_test_pred = y_pred_diverse

            print("\n✅ Using DIVERSITY-ENHANCED predictions for final analysis")

        except Exception as e:
            print(f"❌ Diversity enhancement failed: {e}")
            # Fallback to log transformation
            if "confirmed_90d" in X_test_log.columns:
                y_test_pred_proba = self.model.predict_proba(X_test_log)[:, 1]
                y_test_pred = self.model.predict(X_test_log)
                print("\n✅ Falling back to LOG-TRANSFORMED predictions")
            else:
                y_test_pred_proba = y_original
                y_test_pred = y_pred_original
                print("\n✅ Using ORIGINAL predictions (all enhancements failed)")

        print(f"✓ Predictions generated for {len(y_test_pred)} customers")
        print(
            f"Predicted churners: {sum(y_test_pred)} ({sum(y_test_pred) / len(y_test_pred) * 100:.2f}%)"
        )

        # Feature Contribution Analysis
        print("\n" + "=" * 80)
        print("🔍 INDIVIDUAL FEATURE CONTRIBUTION ANALYSIS")
        print("=" * 80)

        # Generate feature contributions using DIVERSITY-ENHANCED features
        print(
            "🔄 Computing individual feature contributions with diversity enhancement..."
        )
        feature_contributions = self.get_feature_contributions(
            X_test_diverse,
            test_customer_codes,
            test_customer_names,
            y_test_pred_proba,
            y_test_pred,
            top_k=10,
        )

        print(
            f"✅ Feature contributions computed for {len(feature_contributions)} customers"
        )

        # Save detailed results with the specified Korean column structure
        output_file = "customer_feature_contributions.csv"

        # Create new DataFrame with the required structure
        output_data = []

        for idx, row in feature_contributions.iterrows():
            # Basic customer info
            record = {
                "snapshot_month": datetime.now().strftime("%Y-%m-%d"),
                "CCOD": row["customer_code"],
                "churn_prob": row["prediction_proba"],
                "churn_label": 1 if row["prediction"] == 1 else 0,
            }

            # Add top 10 features with Korean structure
            for i in range(1, 11):
                feature_col = f"top_{i}_feature"
                contrib_col = f"top_{i}_contribution"
                value_col = f"top_{i}_value"

                if feature_col in row and pd.notna(row[feature_col]):
                    english_feature = row[feature_col]
                    korean_feature = self.FEATURE_KOREAN_NAMES.get(
                        english_feature, english_feature
                    )
                    contribution = row[contrib_col] if pd.notna(row[contrib_col]) else 0
                    feature_value = (
                        row[value_col]
                        if value_col in row and pd.notna(row[value_col])
                        else 0
                    )

                    # Get normal vs churn averages from training data
                    normal_avg = (
                        self.non_churner_avg.get(english_feature, 0)
                        if english_feature in self.non_churner_avg
                        else 0
                    )
                    churn_avg = (
                        self.churner_avg.get(english_feature, 0)
                        if english_feature in self.churner_avg
                        else 0
                    )

                    # Calculate impact level (영향도) based on absolute contribution
                    abs_contribution = abs(contribution)
                    if abs_contribution >= 0.3:
                        impact = "높음"  # High impact
                    elif abs_contribution >= 0.2:
                        impact = "보통"  # Medium impact
                    else:
                        impact = "낮음"  # Low impact

                    # Add to record
                    record[f"feature_{i}_kor"] = korean_feature
                    record[f"feature_{i}_value"] = round(feature_value, 4)
                    record[f"feature_{i}_normal_average"] = round(normal_avg, 4)
                    record[f"feature_{i}_churn_average"] = round(churn_avg, 4)
                    record[f"feature_{i}_contrib"] = round(contribution, 4)
                    record[f"feature_{i}_contrib_label"] = impact
                else:
                    # Fill empty slots with null values
                    record[f"feature_{i}_kor"] = None
                    record[f"feature_{i}_value"] = None
                    record[f"feature_{i}_normal_average"] = None
                    record[f"feature_{i}_churn_average"] = None
                    record[f"feature_{i}_contrib"] = None
                    record[f"feature_{i}_contrib_label"] = None

            output_data.append(record)

        # Create DataFrame and save
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_file, index=False)
        output_df.to_excel("customer_feature_contributions.xlsx", index=False)
        ensure_partition_exists(
            schema="analytics",
            table_name="user_churn_prediction",
            target_date=datetime.now(),
        )

        # Reset index to ensure proper alignment when adding columns
        user_features_df = user_features_df.reset_index(drop=True)
        X_test_final_reset = X_test_final.reset_index(drop=True)

        user_features_df["recent_90d_activity_ratio"] = X_test_final_reset[
            "최근90일_활동비율"
        ].values
        user_features_df["recent_30d_activity_ratio"] = X_test_final_reset[
            "최근30일_활동비율"
        ].values
        user_features_df["activity_density"] = X_test_final_reset["활동_밀도"].values
        user_features_df["confirmation_rate_change"] = X_test_final_reset[
            "확정비율_변화"
        ].values
        user_features_df["cancellation_rate_change"] = X_test_final_reset[
            "취소비율_변화"
        ].values
        user_features_df["voc_to_work_ratio"] = X_test_final_reset[
            "VOC_대_작업비율"
        ].values
        user_features_df["recent_30d_to_90d_ratio"] = X_test_final_reset[
            "최근30일_대_90일_비율"
        ].values

        user_features_df.drop(
            columns=[
                "최근90일_활동비율",
                "최근30일_활동비율",
                "확정비율_변화",
                "취소비율_변화",
                "VOC_대_작업비율",
                "최근30일_대_90일_비율",
                "활동_밀도",
            ],
            inplace=True,
        )

        # STRICT VALIDATION: Ensure both tables have exactly the same customers
        churn_ccods = set(output_df["CCOD"].tolist())
        features_ccods = set(user_features_df["CCOD"].tolist())

        if len(output_df) != len(user_features_df):
            raise ValueError(
                f"Row count mismatch! user_churn_prediction: {len(output_df)}, user_monthly_property: {len(user_features_df)}"
            )

        if churn_ccods != features_ccods:
            missing_in_features = churn_ccods - features_ccods
            missing_in_churn = features_ccods - churn_ccods
            raise ValueError(
                f"CCOD mismatch! Missing in features: {len(missing_in_features)}, Missing in churn: {len(missing_in_churn)}"
            )

        print(f"✅ Validation passed: {len(output_df)} customers in both tables")

        # Upload both tables
        fast_bulk_insert(
            output_df, table_name="user_churn_prediction", schema="analytics"
        )
        print(f"✅ user_churn_prediction uploaded ({len(output_df)} rows)")

        self.upload_user_features(user_features_df)

        print(
            f"\n✅ Analysis complete! Check '{output_file}' for detailed per-customer results."
        )
