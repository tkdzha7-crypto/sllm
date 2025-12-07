import os
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class ChurnModelTrainer:
    """Trainer class for customer churn prediction model"""

    def __init__(self, output_dir="src/models", test_size=0.2, random_state=42):
        """
        Initialize the trainer

        Args:
            output_dir: Directory to save trained models
            test_size: Proportion of dataset to include in test split
            random_state: Random state for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        self.random_state = random_state

        self.model = None
        self.feature_names = None
        self.churner_avg = None
        self.non_churner_avg = None

    def get_db_engine(self):
        """Create database engine for data extraction"""
        db_user = os.getenv("DB_USER", "cesco_admin")
        db_pass = os.getenv("DB_PASS", "Cesco_1588")
        db_host = os.getenv("DB_HOST", "localhost")
        db_name = os.getenv("DB_NAME", "deskroom_core")
        db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:5432/{db_name}"
        return create_engine(db_url)

    def load_training_data(self, churn_lookback_days=90, min_work_history=5):
        """
        Load and prepare training data from database

        Args:
            churn_lookback_days: Number of days to look back for churn definition
            min_work_history: Minimum work records required for inclusion

        Returns:
            DataFrame with features and churn labels
        """
        print("🔄 Loading training data from database...")
        engine = self.get_db_engine()

        # Query to get user features from user_monthly_property table
        # This assumes the table already has historical data
        query = """
        SELECT
            ccod,
            created_at,
            unique_work_types,
            cancelled_work,
            cancellation_rate,
            confirmation_rate,
            avg_services_per_work,
            work_last_30d,
            confirmed_30d,
            cancelled_30d,
            services_30d,
            work_types_30d,
            cancellation_rate_30d,
            confirmation_rate_30d,
            work_last_60d,
            confirmed_60d,
            cancelled_60d,
            services_60d,
            work_types_60d,
            cancellation_rate_60d,
            confirmation_rate_60d,
            work_last_90d,
            confirmed_90d,
            cancelled_90d,
            services_90d,
            work_types_90d,
            cancellation_rate_90d,
            confirmation_rate_90d,
            avg_csi_score,
            csi_score_30d,
            csi_score_60d,
            csi_score_90d,
            csi_survey_count,
            csi_survey_count_30d,
            csi_survey_count_60d,
            csi_survey_count_90d,
            num_interactions,
            voc_count,
            non_voc_count,
            voc_ratio,
            interactions_30d,
            voc_30d,
            non_voc_30d,
            voc_ratio_30d,
            interactions_60d,
            voc_60d,
            non_voc_60d,
            voc_ratio_60d,
            interactions_90d,
            voc_90d,
            non_voc_90d,
            voc_ratio_90d,
            recent_90d_activity_ratio,
            recent_30d_activity_ratio,
            activity_density,
            confirmation_rate_change,
            cancellation_rate_change,
            voc_to_work_ratio,
            recent_30d_to_90d_ratio
        FROM analytics.user_monthly_property
        WHERE work_last_90d >= :min_work
        ORDER BY created_at DESC
        """

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"min_work": min_work_history})

        print(f"✅ Loaded {len(df)} records from database")

        # Define churn label (this is a simplified approach)
        # In production, you'd have actual churn labels from business logic
        # Here we'll use a heuristic: customers with very low recent activity are "churned"
        df["churn"] = (
            (df["work_last_30d"] == 0)
            & (df["work_last_60d"] <= 1)
            & (df["work_last_90d"] <= 2)
        ).astype(int)

        print("📊 Churn distribution:")
        print(df["churn"].value_counts())
        print(f"   Churn rate: {df['churn'].mean():.2%}")

        return df

    def prepare_features(self, df):
        """
        Prepare feature matrix and target variable

        Args:
            df: DataFrame with features and churn labels

        Returns:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
        """
        # Select feature columns (exclude metadata columns)
        feature_cols = [
            col for col in df.columns if col not in ["ccod", "created_at", "churn"]
        ]

        X = df[feature_cols].copy()
        y = df["churn"].copy()

        # Handle missing values
        X = X.fillna(0)

        # Store feature names
        self.feature_names = X.columns.tolist()

        print(f"✅ Prepared {len(self.feature_names)} features")
        return X, y

    def train_model(
        self, X_train, y_train, X_val=None, y_val=None, use_grid_search=False
    ):
        """
        Train XGBoost model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            use_grid_search: Whether to use GridSearchCV for hyperparameter tuning
        """
        print("🔄 Training XGBoost model...")

        if use_grid_search:
            print("🔍 Performing grid search for hyperparameter tuning...")

            # Define parameter grid
            param_grid = {
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "n_estimators": [100, 200, 300],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "scale_pos_weight": [1, 3, 5],  # For imbalanced data
            }

            xgb_model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=self.random_state,
                use_label_encoder=False,
            )

            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
            )

            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            print(f"✅ Best parameters: {grid_search.best_params_}")
            print(f"✅ Best CV score: {grid_search.best_score_:.4f}")

        else:
            # Use default parameters with some imbalance handling
            self.model = xgb.XGBClassifier(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=3,  # Adjust based on class imbalance
                objective="binary:logistic",
                eval_metric="auc",
                random_state=self.random_state,
                use_label_encoder=False,
            )

            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_train), (X_val, y_val)]
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=20,
                    verbose=False,
                )
            else:
                self.model.fit(X_train, y_train)

            print("✅ Model training completed")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Test labels
        """
        print("\n📊 Model Evaluation:")
        print("=" * 60)

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Classification report
        print("\nClassification Report:")
        print(
            classification_report(
                y_test, y_pred, target_names=["Not Churned", "Churned"]
            )
        )

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # ROC AUC score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\n🎯 ROC AUC Score: {roc_auc:.4f}")

        # Feature importance
        print("\n🔝 Top 10 Most Important Features:")
        feature_importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print(feature_importance.head(10).to_string(index=False))

        # Plot ROC curve
        self._plot_roc_curve(y_test, y_pred_proba, roc_auc)

        # Plot feature importance
        self._plot_feature_importance(feature_importance)

        return {
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "feature_importance": feature_importance,
        }

    def _plot_roc_curve(self, y_test, y_pred_proba, roc_auc):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Churn Prediction Model")
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = self.output_dir / "churn_prediction" / "roc_curve.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ ROC curve saved to {output_path}")
        plt.close()

    def _plot_feature_importance(self, feature_importance, top_n=20):
        """Plot feature importance"""
        top_features = feature_importance.head(top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importance - Churn Prediction")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        output_path = self.output_dir / "churn_prediction" / "feature_importance.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Feature importance plot saved to {output_path}")
        plt.close()

    def compute_training_averages(self, X_train, y_train):
        """
        Compute average feature values for churned and non-churned customers

        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("🔄 Computing training averages...")

        X_train_df = pd.DataFrame(X_train, columns=self.feature_names)

        # Compute averages for churned customers
        self.churner_avg = X_train_df[y_train == 1].mean()

        # Compute averages for non-churned customers
        self.non_churner_avg = X_train_df[y_train == 0].mean()

        print("✅ Training averages computed")

    def save_model(self):
        """Save trained model and artifacts"""
        print("💾 Saving model and artifacts...")

        # Save model
        model_path = self.output_dir / "churn_model_with_csi.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"✅ Model saved to {model_path}")

        # Save training averages
        averages = {
            "churner_avg": self.churner_avg,
            "non_churner_avg": self.non_churner_avg,
            "feature_names": self.feature_names,
        }

        averages_path = self.output_dir / "training_averages_with_csi.pkl"
        with open(averages_path, "wb") as f:
            pickle.dump(averages, f)
        print(f"✅ Training averages saved to {averages_path}")

    def run_full_training_pipeline(self, use_grid_search=False):
        """
        Run the complete training pipeline

        Args:
            use_grid_search: Whether to use GridSearchCV
        """
        print("\n" + "=" * 60)
        print("🚀 Starting Churn Prediction Model Training Pipeline")
        print("=" * 60 + "\n")

        # Load data
        df = self.load_training_data()

        # Prepare features
        X, y = self.prepare_features(df)

        # Split data
        print(f"🔄 Splitting data (test_size={self.test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        print(f"✅ Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Further split training into train/validation if not using grid search
        if not use_grid_search:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y_train,
            )
        else:
            X_val, y_val = None, None

        # Train model
        self.train_model(X_train, y_train, X_val, y_val, use_grid_search)

        # Compute training averages
        self.compute_training_averages(X_train, y_train)

        # Evaluate model
        eval_results = self.evaluate_model(X_test, y_test)

        # Save model
        self.save_model()

        print("\n" + "=" * 60)
        print("✅ Training pipeline completed successfully!")
        print("=" * 60 + "\n")

        return eval_results


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description="Train Churn Prediction Model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set size (default: 0.2)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Use GridSearchCV for hyperparameter tuning",
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = ChurnModelTrainer(
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Run training pipeline
    trainer.run_full_training_pipeline(use_grid_search=args.grid_search)


if __name__ == "__main__":
    main()
