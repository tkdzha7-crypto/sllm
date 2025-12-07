import os
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.recommendation.category_mapper import load_category_mapping
from src.models.recommendation.mapping_constants import 업태_TO_분류, 종목_TO_분류


class RecommendationModelTrainer:
    """Trainer class for user clustering-based recommendation model"""

    def __init__(
        self, output_dir="src/models", n_clusters=10, pca_components=2, random_state=42
    ):
        """
        Initialize the trainer

        Args:
            output_dir: Directory to save trained models
            n_clusters: Number of clusters for K-Means
            pca_components: Number of PCA components for business classification
            random_state: Random state for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.random_state = random_state

        # Model components
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

        # Mapping constants
        self.업태_to_분류 = 업태_TO_분류
        self.종목_to_분류 = 종목_TO_분류
        self.category_mapping = load_category_mapping()

        # Feature columns
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

        # Results
        self.clustering_df = None
        self.cluster_recommendations = None

    def get_db_engine(self):
        """Create database engine for data extraction"""
        db_user = os.getenv("DB_USER", "cesco_admin")
        db_pass = os.getenv("DB_PASS", "Cesco_1588")
        db_host = os.getenv("DB_HOST", "localhost")
        db_name = os.getenv("DB_NAME", "deskroom_core")
        db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:5432/{db_name}"
        return create_engine(db_url)

    def load_training_data(self):
        """
        Load and prepare training data from database

        Returns:
            DataFrame with user information for clustering
        """
        print("🔄 Loading training data from database...")
        engine = self.get_db_engine()

        # Query to get user features from user_monthly_features
        # This should be adapted to your actual table structure
        query = """
        SELECT DISTINCT
            ccod,
            user_information,
            contract_info,
            purchase_logs
        FROM source.user_monthly_features
        WHERE user_information IS NOT NULL
        ORDER BY ccod
        LIMIT 10000  -- Adjust based on your data size
        """

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        print(f"✅ Loaded {len(df)} user records from database")

        return df

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

        sido = RecommendationModelTrainer.extract_sido(address)
        if sido:
            address = address.replace(sido, "").strip()

        parts = address.split()
        if len(parts) > 0:
            sigungu = parts[0]
            if "(" in sigungu:
                sigungu = sigungu.split("(")[0].strip()
            return sigungu
        return None

    def prepare_features(self, df):
        """
        Prepare feature matrix for clustering

        Args:
            df: DataFrame with user information

        Returns:
            Prepared DataFrame with encoded features
        """
        print("🔄 Preparing features for clustering...")

        # Parse JSON columns
        import json

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

        # Extract features from user_information
        df["주소"] = df["user_information"].apply(
            lambda x: x.get("주소") if isinstance(x, dict) else None
        )
        df["위도"] = df["user_information"].apply(
            lambda x: float(x.get("위도", 0))
            if isinstance(x, dict) and x.get("위도")
            else 37.5665
        )
        df["경도"] = df["user_information"].apply(
            lambda x: float(x.get("경도", 0))
            if isinstance(x, dict) and x.get("경도")
            else 126.9780
        )
        df["업태"] = df["user_information"].apply(
            lambda x: x.get("업태") if isinstance(x, dict) else None
        )
        df["종목"] = df["user_information"].apply(
            lambda x: x.get("종목") if isinstance(x, dict) else None
        )
        df["평균_면적"] = df["user_information"].apply(
            lambda x: float(x.get("평균_면적", 0))
            if isinstance(x, dict) and x.get("평균_면적")
            else 0
        )

        # Extract geographic features
        df["시도"] = df["주소"].apply(self.extract_sido)
        df["시군구"] = df["주소"].apply(self.extract_sigungu)

        # Map 업태 and 종목 to classification
        df["대분류"] = df["업태"].map(self.업태_to_분류)
        df["중분류"] = df["종목"].map(self.종목_to_분류)
        df["소분류"] = None  # Would need additional mapping
        df["세분류"] = None  # Would need additional mapping

        # Fill missing values
        df["시도"] = df["시도"].fillna("Unknown")
        df["시군구"] = df["시군구"].fillna("Unknown")
        df["대분류"] = df["대분류"].fillna("기타")
        df["중분류"] = df["중분류"].fillna("기타")
        df["소분류"] = df["소분류"].fillna("기타")
        df["세분류"] = df["세분류"].fillna("기타")
        df["업태"] = df["업태"].fillna("기타")

        # Categorize 평균_면적
        df["평균_면적_category"] = pd.cut(
            df["평균_면적"], bins=[0, 50, 100, 200, 500, np.inf], labels=[0, 1, 2, 3, 4]
        ).astype(int)

        # Encode categorical features
        print("🔄 Encoding categorical features...")
        self.le_sido = LabelEncoder()
        self.le_sigungu = LabelEncoder()
        self.le_대분류 = LabelEncoder()
        self.le_중분류 = LabelEncoder()
        self.le_소분류 = LabelEncoder()
        self.le_세분류 = LabelEncoder()
        self.le_업태 = LabelEncoder()

        df["시도_encoded"] = self.le_sido.fit_transform(df["시도"])
        df["시군구_encoded"] = self.le_sigungu.fit_transform(df["시군구"])
        df["대분류_encoded"] = self.le_대분류.fit_transform(df["대분류"])
        df["중분류_encoded"] = self.le_중분류.fit_transform(df["중분류"])
        df["소분류_encoded"] = self.le_소분류.fit_transform(df["소분류"])
        df["세분류_encoded"] = self.le_세분류.fit_transform(df["세분류"])
        df["업태_encoded"] = self.le_업태.fit_transform(df["업태"])

        # Apply PCA to classification features
        print("🔄 Applying PCA to classification features...")
        classification_features = df[
            ["대분류_encoded", "중분류_encoded", "소분류_encoded", "세분류_encoded"]
        ].values

        self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        pca_result = self.pca.fit_transform(classification_features)

        df["분류_PCA1"] = pca_result[:, 0]
        df["분류_PCA2"] = pca_result[:, 1]

        print(f"✅ PCA explained variance ratio: {self.pca.explained_variance_ratio_}")

        # Select final features
        feature_df = df[
            ["ccod", "user_information", "contract_info", "purchase_logs"]
            + self.feature_cols
        ].copy()

        print(
            f"✅ Features prepared: {len(feature_df)} samples, {len(self.feature_cols)} features"
        )

        return feature_df

    def train_clustering_model(self, X, optimize_k=False, k_range=None):
        """
        Train K-Means clustering model

        Args:
            X: Feature matrix
            optimize_k: Whether to find optimal number of clusters
            k_range: Range of k values to test if optimize_k is True
        """
        print("🔄 Training clustering model...")

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if optimize_k:
            print("🔍 Finding optimal number of clusters...")
            if k_range is None:
                k_range = range(2, 21)

            inertias = []
            silhouette_scores = []
            davies_bouldin_scores = []

            for k in k_range:
                print(f"  Testing k={k}...", end=" ")
                kmeans_temp = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    n_init=10,
                    max_iter=300,
                )
                labels = kmeans_temp.fit_predict(X_scaled)

                inertias.append(kmeans_temp.inertia_)
                silhouette_scores.append(silhouette_score(X_scaled, labels))
                davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))
                print(f"Silhouette: {silhouette_scores[-1]:.3f}")

            # Plot evaluation metrics
            self._plot_cluster_evaluation(
                k_range, inertias, silhouette_scores, davies_bouldin_scores
            )

            # Select best k based on silhouette score
            best_k_idx = np.argmax(silhouette_scores)
            best_k = list(k_range)[best_k_idx]
            print(f"✅ Optimal number of clusters: {best_k}")
            self.n_clusters = best_k

        # Train final model
        print(f"🔄 Training K-Means with k={self.n_clusters}...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
        )
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # Evaluate
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)

        print("✅ Clustering completed")
        print(f"   Silhouette Score: {silhouette_avg:.4f}")
        print(f"   Davies-Bouldin Index: {davies_bouldin:.4f}")

        return cluster_labels

    def _plot_cluster_evaluation(
        self, k_range, inertias, silhouette_scores, davies_bouldin_scores
    ):
        """Plot cluster evaluation metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Elbow plot
        axes[0].plot(k_range, inertias, "bo-")
        axes[0].set_xlabel("Number of Clusters (k)")
        axes[0].set_ylabel("Inertia")
        axes[0].set_title("Elbow Method")
        axes[0].grid(True, alpha=0.3)

        # Silhouette score
        axes[1].plot(k_range, silhouette_scores, "ro-")
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette Score (Higher is Better)")
        axes[1].grid(True, alpha=0.3)

        # Davies-Bouldin Index
        axes[2].plot(k_range, davies_bouldin_scores, "go-")
        axes[2].set_xlabel("Number of Clusters (k)")
        axes[2].set_ylabel("Davies-Bouldin Index")
        axes[2].set_title("Davies-Bouldin Index (Lower is Better)")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / "recommendation" / "cluster_evaluation.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Cluster evaluation plot saved to {output_path}")
        plt.close()

    def analyze_clusters(self, feature_df, cluster_labels):
        """
        Analyze cluster characteristics and generate recommendations

        Args:
            feature_df: DataFrame with features and user info
            cluster_labels: Cluster assignments
        """
        print("🔄 Analyzing clusters...")

        # Add cluster labels
        feature_df["cluster"] = cluster_labels
        self.clustering_df = feature_df.copy()

        # Parse JSON if needed
        import json

        def safe_json_loads(x):
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except json.JSONDecodeError:
                    return None
            return x

        # Analyze each cluster
        cluster_recommendations = {}

        for cluster_id in range(self.n_clusters):
            cluster_data = feature_df[feature_df["cluster"] == cluster_id]

            print(f"\n📊 Cluster {cluster_id}: {len(cluster_data)} users")

            # Aggregate contract and purchase data
            all_contracts = []
            all_products = []

            for _, row in cluster_data.iterrows():
                contracts = row["contract_info"]
                purchases = row["purchase_logs"]

                if isinstance(contracts, list):
                    for contract in contracts:
                        if isinstance(contract, dict) and "계약대상" in contract:
                            all_contracts.append(contract["계약대상"])

                if isinstance(purchases, list):
                    for purchase in purchases:
                        if isinstance(purchase, dict) and "service_name" in purchase:
                            all_products.append(purchase["service_name"])

            # Find top contracts and products
            from collections import Counter

            contract_counts = Counter(all_contracts)
            product_counts = Counter(all_products)

            top_contracts = contract_counts.most_common(5)
            top_products = product_counts.most_common(5)

            cluster_recommendations[cluster_id] = {
                "size": len(cluster_data),
                "top_contracts": [
                    {
                        "code": code,
                        "count": count,
                        "percentage": count / len(cluster_data) * 100,
                    }
                    for code, count in top_contracts
                ],
                "top_products": [
                    {
                        "name": name,
                        "count": count,
                        "percentage": count / len(cluster_data) * 100,
                    }
                    for name, count in top_products
                ],
            }

            print(
                f"   Top contracts: {[c['code'] for c in cluster_recommendations[cluster_id]['top_contracts'][:3]]}"
            )
            print(
                f"   Top products: {[p['name'] for p in cluster_recommendations[cluster_id]['top_products'][:3]]}"
            )

        self.cluster_recommendations = cluster_recommendations
        print("\n✅ Cluster analysis completed")

    def visualize_clusters(self, X, cluster_labels):
        """Visualize clusters using PCA"""
        print("🔄 Visualizing clusters...")

        # Apply PCA for visualization (2D)
        pca_vis = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca_vis.fit_transform(X)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", alpha=0.6, s=50
        )
        plt.colorbar(scatter, label="Cluster")
        plt.xlabel(f"PC1 ({pca_vis.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca_vis.explained_variance_ratio_[1]:.2%} variance)")
        plt.title("User Clusters Visualization (PCA)")
        plt.grid(True, alpha=0.3)

        output_path = self.output_dir / "recommendation" / "cluster_visualization.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Cluster visualization saved to {output_path}")
        plt.close()

    def save_model(self):
        """Save trained model and artifacts"""
        print("💾 Saving model and artifacts...")

        artifacts = {
            "kmeans": self.kmeans,
            "scaler": self.scaler,
            "pca": self.pca,
            "le_sido": self.le_sido,
            "le_sigungu": self.le_sigungu,
            "le_대분류": self.le_대분류,
            "le_중분류": self.le_중분류,
            "le_소분류": self.le_소분류,
            "le_세분류": self.le_세분류,
            "le_업태": self.le_업태,
            "cluster_recommendations": self.cluster_recommendations,
            "clustering_df": self.clustering_df,
        }

        model_path = self.output_dir / "simple_clustering_model_pca.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(artifacts, f)

        print(f"✅ Model saved to {model_path}")

    def run_full_training_pipeline(self, optimize_k=False):
        """
        Run the complete training pipeline

        Args:
            optimize_k: Whether to find optimal number of clusters
        """
        print("\n" + "=" * 60)
        print("🚀 Starting Recommendation Model Training Pipeline")
        print("=" * 60 + "\n")

        # Load data
        df = self.load_training_data()

        # Prepare features
        feature_df = self.prepare_features(df)

        # Extract feature matrix
        X = feature_df[self.feature_cols].values

        # Train clustering model
        cluster_labels = self.train_clustering_model(X, optimize_k=optimize_k)

        # Analyze clusters
        self.analyze_clusters(feature_df, cluster_labels)

        # Visualize clusters
        self.visualize_clusters(X, cluster_labels)

        # Save model
        self.save_model()

        print("\n" + "=" * 60)
        print("✅ Training pipeline completed successfully!")
        print("=" * 60 + "\n")


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description="Train Recommendation Model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=10, help="Number of clusters (default: 10)"
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=2,
        help="Number of PCA components (default: 2)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )
    parser.add_argument(
        "--optimize-k", action="store_true", help="Find optimal number of clusters"
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = RecommendationModelTrainer(
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        pca_components=args.pca_components,
        random_state=args.random_state,
    )

    # Run training pipeline
    trainer.run_full_training_pipeline(optimize_k=args.optimize_k)


if __name__ == "__main__":
    main()
