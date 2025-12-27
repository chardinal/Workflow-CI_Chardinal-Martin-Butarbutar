"""
Retail Transaction Clustering - MLflow Training Pipeline
Author: Chardinal Martin Butarbutar
Description: KMeans clustering model untuk segmentasi transaksi retail
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CONFIGURATION
# ===============================
DATASET_PATH = "Retail_Transaction_Dataset_preprocessing.csv"
N_CLUSTERS = 4
RANDOM_STATE = 42
ARTIFACTS_DIR = "artifacts"

# ===============================
# HEADER
# ===============================
print("=" * 70)
print("🚀 MLflow Retail Transaction Clustering - Training Pipeline")
print("=" * 70)
print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📦 MLflow Version: {mlflow.__version__}")
print("=" * 70)

# ===============================
# 1. LOAD DATASET
# ===============================
print("\n" + "=" * 70)
print("📂 STEP 1: Loading Dataset")
print("=" * 70)

try:
    # Check if file exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset '{DATASET_PATH}' not found!")
    
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\n📋 Column Information:")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n📈 Dataset Info:")
    print(df.info())
    
    print(f"\n📊 Statistical Summary:")
    print(df.describe())
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  Warning: Missing values detected:")
        print(missing[missing > 0])
    else:
        print(f"\n✅ No missing values detected")
    
except FileNotFoundError as e:
    print(f"❌ Error: {str(e)}")
    print(f"📂 Current directory: {os.getcwd()}")
    print(f"📂 Files available: {os.listdir('.')}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error loading dataset: {str(e)}")
    sys.exit(1)

# ===============================
# 2. FEATURE SELECTION
# ===============================
print("\n" + "=" * 70)
print("🔍 STEP 2: Feature Selection")
print("=" * 70)

# Select only numeric features
X = df.select_dtypes(include=["int64", "float64", "int32", "float32"])

print(f"✅ Selected {X.shape[1]} numeric features:")
for i, col in enumerate(X.columns, 1):
    print(f"   {i}. {col:25s} - dtype: {X[col].dtype}")

# Handle missing values if any
if X.isnull().sum().sum() > 0:
    print(f"\n⚠️  Handling missing values...")
    X = X.fillna(X.mean())
    print(f"✅ Missing values filled with column means")

print(f"\n📊 Feature Matrix Shape: {X.shape}")

# ===============================
# 3. DATA STANDARDIZATION
# ===============================
print("\n" + "=" * 70)
print("📐 STEP 3: Data Standardization")
print("=" * 70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✅ Data standardized using StandardScaler")
print(f"   Original mean: {X.mean().mean():.2f}")
print(f"   Scaled mean: {X_scaled.mean():.6f} (target: ~0)")
print(f"   Original std: {X.std().mean():.2f}")
print(f"   Scaled std: {X_scaled.std():.6f} (target: ~1)")

# ===============================
# 4. MLFLOW EXPERIMENT SETUP
# ===============================
print("\n" + "=" * 70)
print("🧪 STEP 4: MLflow Experiment Setup")
print("=" * 70)

# Set experiment - this is handled by mlflow run command
experiment_name = "Retail_Transaction_Clustering"
print(f"✅ Experiment Name: '{experiment_name}'")

# Get or use active run (created by mlflow run command)
active_run = mlflow.active_run()
if active_run:
    print(f"🆔 Using Active MLflow Run ID: {active_run.info.run_id}")
    run_id = active_run.info.run_id
else:
    print(f"⚠️  No active run found, will create new run")
    run_id = None

# ===============================
# 5. MODEL TRAINING
# ===============================
print("\n" + "=" * 70)
print("🤖 STEP 5: Model Training - KMeans Clustering")
print("=" * 70)

print(f"\n⚙️  Model Parameters:")
print(f"   • Algorithm: KMeans")
print(f"   • Number of Clusters: {N_CLUSTERS}")
print(f"   • Random State: {RANDOM_STATE}")
print(f"   • Initialization: k-means++")
print(f"   • Max Iterations: 300")
print(f"   • Number of Initializations: 10")

# Initialize and train model
model = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10,
    max_iter=300,
    init='k-means++'
)

print(f"\n🔄 Training model...")
labels = model.fit_predict(X_scaled)
print(f"✅ Model training completed!")
print(f"✅ Convergence achieved in {model.n_iter_} iterations")

# ===============================
# 6. MODEL EVALUATION
# ===============================
print("\n" + "=" * 70)
print("📊 STEP 6: Model Evaluation")
print("=" * 70)

# Calculate evaluation metrics
silhouette = silhouette_score(X_scaled, labels)
davies_bouldin = davies_bouldin_score(X_scaled, labels)
calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
inertia = model.inertia_

print(f"\n🎯 Clustering Quality Metrics:")
print(f"   • Silhouette Score: {silhouette:.4f}")
print(f"     └─ Range: -1 to 1 (higher is better)")
print(f"     └─ Interpretation: {'Excellent' if silhouette > 0.7 else 'Good' if silhouette > 0.5 else 'Fair' if silhouette > 0.3 else 'Poor'}")

print(f"\n   • Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"     └─ Range: 0 to ∞ (lower is better)")
print(f"     └─ Interpretation: {'Excellent' if davies_bouldin < 0.5 else 'Good' if davies_bouldin < 1.0 else 'Fair' if davies_bouldin < 2.0 else 'Poor'}")

print(f"\n   • Calinski-Harabasz Score: {calinski_harabasz:.2f}")
print(f"     └─ Range: 0 to ∞ (higher is better)")
print(f"     └─ Interpretation: {'Excellent' if calinski_harabasz > 1000 else 'Good' if calinski_harabasz > 500 else 'Fair' if calinski_harabasz > 100 else 'Poor'}")

print(f"\n   • Inertia (Within-cluster sum of squares): {inertia:.2f}")

# Cluster distribution analysis
print(f"\n📈 Cluster Distribution Analysis:")
cluster_counts = pd.Series(labels).value_counts().sort_index()

for cluster_id in range(N_CLUSTERS):
    count = cluster_counts.get(cluster_id, 0)
    percentage = (count / len(labels)) * 100
    bar_length = int(percentage / 2)
    bar = "█" * bar_length
    print(f"   Cluster {cluster_id}: {count:6,} samples ({percentage:5.1f}%) {bar}")

# Check for imbalanced clusters
max_cluster_size = cluster_counts.max()
min_cluster_size = cluster_counts.min()
balance_ratio = max_cluster_size / min_cluster_size

if balance_ratio > 5:
    print(f"\n⚠️  Warning: Clusters are highly imbalanced (ratio: {balance_ratio:.1f}:1)")
elif balance_ratio > 3:
    print(f"\n⚠️  Note: Moderate cluster imbalance detected (ratio: {balance_ratio:.1f}:1)")
else:
    print(f"\n✅ Clusters are well-balanced (ratio: {balance_ratio:.1f}:1)")

# ===============================
# 7. MLFLOW LOGGING
# ===============================
print("\n" + "=" * 70)
print("📝 STEP 7: Logging to MLflow")
print("=" * 70)

# Log parameters
mlflow.log_param("algorithm", "KMeans")
mlflow.log_param("n_clusters", N_CLUSTERS)
mlflow.log_param("random_state", RANDOM_STATE)
mlflow.log_param("n_features", X.shape[1])
mlflow.log_param("n_samples", X.shape[0])
mlflow.log_param("n_iterations", model.n_iter_)
mlflow.log_param("initialization_method", "k-means++")
mlflow.log_param("max_iterations", 300)

print(f"✅ Parameters logged:")
print(f"   • Algorithm, n_clusters, random_state")
print(f"   • Dataset dimensions, iterations")

# Log metrics
mlflow.log_metric("silhouette_score", silhouette)
mlflow.log_metric("davies_bouldin_index", davies_bouldin)
mlflow.log_metric("calinski_harabasz_score", calinski_harabasz)
mlflow.log_metric("inertia", inertia)
mlflow.log_metric("balance_ratio", float(balance_ratio))

print(f"✅ Metrics logged:")
print(f"   • silhouette_score, davies_bouldin_index")
print(f"   • calinski_harabasz_score, inertia")

# Log cluster sizes
for cluster_id in range(N_CLUSTERS):
    count = cluster_counts.get(cluster_id, 0)
    mlflow.log_metric(f"cluster_{cluster_id}_size", count)
    mlflow.log_metric(f"cluster_{cluster_id}_percentage", (count / len(labels)) * 100)

print(f"✅ Cluster distribution logged")

# ===============================
# 8. SAVE ARTIFACTS
# ===============================
print("\n" + "=" * 70)
print("💾 STEP 8: Saving Model Artifacts")
print("=" * 70)

# Create artifacts directory
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 8.1 Save model
model_path = os.path.join(ARTIFACTS_DIR, "kmeans_model.pkl")
joblib.dump(model, model_path)
model_size = os.path.getsize(model_path) / 1024  # KB
print(f"✅ Model saved: {model_path} ({model_size:.1f} KB)")

# 8.2 Save scaler
scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
scaler_size = os.path.getsize(scaler_path) / 1024  # KB
print(f"✅ Scaler saved: {scaler_path} ({scaler_size:.1f} KB)")

# 8.3 Save cluster labels
labels_df = pd.DataFrame({
    'cluster_label': labels,
    'original_index': range(len(labels))
})
labels_path = os.path.join(ARTIFACTS_DIR, "cluster_labels.csv")
labels_df.to_csv(labels_path, index=False)
print(f"✅ Cluster labels saved: {labels_path}")

# 8.4 Save cluster centers
centers_df = pd.DataFrame(
    model.cluster_centers_,
    columns=X.columns
)
centers_path = os.path.join(ARTIFACTS_DIR, "cluster_centers.csv")
centers_df.to_csv(centers_path, index=False)
print(f"✅ Cluster centers saved: {centers_path}")

# 8.5 Save feature names
feature_names_path = os.path.join(ARTIFACTS_DIR, "feature_names.txt")
with open(feature_names_path, 'w') as f:
    f.write("\n".join(X.columns))
print(f"✅ Feature names saved: {feature_names_path}")

# 8.6 Save metadata
metadata = {
    'run_id': run_id if run_id else 'local_run',
    'experiment_name': experiment_name,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'algorithm': 'KMeans',
    'n_clusters': N_CLUSTERS,
    'random_state': RANDOM_STATE,
    'n_samples': X.shape[0],
    'n_features': X.shape[1],
    'n_iterations': model.n_iter_,
    'silhouette_score': float(silhouette),
    'davies_bouldin_index': float(davies_bouldin),
    'calinski_harabasz_score': float(calinski_harabasz),
    'inertia': float(inertia),
    'balance_ratio': float(balance_ratio)
}

metadata_path = os.path.join(ARTIFACTS_DIR, "metadata.txt")
with open(metadata_path, 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("MODEL TRAINING METADATA\n")
    f.write("=" * 50 + "\n\n")
    for key, value in metadata.items():
        f.write(f"{key}: {value}\n")
    f.write("\n" + "=" * 50 + "\n")
    f.write("CLUSTER DISTRIBUTION\n")
    f.write("=" * 50 + "\n")
    for cluster_id in range(N_CLUSTERS):
        count = cluster_counts.get(cluster_id, 0)
        percentage = (count / len(labels)) * 100
        f.write(f"Cluster {cluster_id}: {count} samples ({percentage:.1f}%)\n")

print(f"✅ Metadata saved: {metadata_path}")

# 8.7 Log all artifacts to MLflow
print(f"\n📤 Uploading artifacts to MLflow...")
mlflow.log_artifact(model_path)
mlflow.log_artifact(scaler_path)
mlflow.log_artifact(labels_path)
mlflow.log_artifact(centers_path)
mlflow.log_artifact(feature_names_path)
mlflow.log_artifact(metadata_path)

# Log model with MLflow
mlflow.sklearn.log_model(
    model, 
    "model",
    conda_env={
        'name': 'mlflow-env',
        'channels': ['defaults', 'conda-forge'],
        'dependencies': [
            'python=3.9',
            'pip',
            {'pip': [
                'mlflow==2.10.0',
                'scikit-learn==1.4.0',
                'pandas==2.2.0',
                'joblib==1.3.2'
            ]}
        ]
    }
)

print(f"✅ All artifacts uploaded to MLflow successfully!")

# ===============================
# 9. TRAINING SUMMARY
# ===============================
print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)

print(f"\n📊 Final Results Summary:")
print(f"{'─' * 70}")
print(f"   Model Algorithm       : KMeans Clustering")
print(f"   Number of Clusters    : {N_CLUSTERS}")
print(f"   Total Samples         : {X.shape[0]:,}")
print(f"   Number of Features    : {X.shape[1]}")
print(f"   Convergence Iterations: {model.n_iter_}")
print(f"{'─' * 70}")
print(f"   Silhouette Score      : {silhouette:.4f}")
print(f"   Davies-Bouldin Index  : {davies_bouldin:.4f}")
print(f"   Calinski-Harabasz     : {calinski_harabasz:.2f}")
print(f"   Inertia (WCSS)        : {inertia:.2f}")
print(f"   Balance Ratio         : {balance_ratio:.2f}:1")
print(f"{'─' * 70}")

print(f"\n📦 Artifacts Location:")
print(f"   Local Directory : ./{ARTIFACTS_DIR}/")

print(f"\n⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print("✅ All processes completed successfully!")
print("✅ Model is ready for deployment!")
print("=" * 70)
