# 🛡️ CICIDS2017 Network Intrusion Detection Engine (PySpark & Streamlit)

A production-grade, distributed big data mini project for **Network Intrusion Detection System (NIDS)** analyzing over 2.8 million network flow records from the **CICIDS2017** benchmark dataset using **Apache Spark (PySpark)**, **Spark MLlib**, and an interactive, dark-themed **Streamlit + Plotly** analytical dashboard.

---

## 🏗️ Architecture & Data Flow

```mermaid
flowchart TD
    subgraph INGESTION ["1. Ingestion Layer"]
        A["data/raw/*.csv\n(8 Files, ~2.8M Flows)"] -->|Explicit Static Schema| B["src/ingest.py\nSchema Validation & Typing"]
        B -->|Add day, source_file, simulate hours| C[("data/parquet/raw_flows.parquet\nPartitioned by 'day'")]
    end

    subgraph CLEANING ["2. Data Cleaning Layer"]
        C --> D["src/clean.py"]
        D -->|Drop Inf / Nulls / Duplicates| E["Zero-Variance Pruning"]
        E -->|Web Attack Consolidation| F[("data/parquet/cleaned_flows.parquet")]
    end

    subgraph EDA ["3. Exploratory Data Analysis"]
        F --> G["src/eda.py"]
        G --> H["Class Imbalance Breakdown"]
        G --> I["Temporal Day/Hour Aggregates"]
        G --> J["Pearson Correlation Matrix"]
    end

    subgraph FEATURES ["4. Feature Engineering & Selection"]
        F --> K["src/features.py"]
        K -->|Train/Test Stratified Split 80/20| L["Train / Test Partitions"]
        L -->|Prune Multicollinearity |r| > 0.95| M["Non-Collinear Features"]
        M -->|Random Forest Importance| N["Top-30 Ranked Features"]
        N -->|StringIndexer + VectorAssembler + StandardScaler| O["Spark ML Pipeline Model"]
        L -->|Class Weighting / Benign Downsample| P[("data/parquet/train_features.parquet\ndata/parquet/test_features.parquet")]
    end

    subgraph ML_EVAL ["5. Model Training & Evaluation"]
        P --> Q["src/train.py"]
        Q --> R["Decision Tree (Baseline)"]
        Q --> S["Random Forest (Bagging)"]
        Q --> T["Logistic Regression (Multinomial)"]
        Q --> U["GBT (One-vs-Rest, Opt)"]
        R & S & T & U --> V["Macro-F1 Model Selection"]
        V --> W[("data/output/models/best_model\ndata/output/metrics/*.json")]
    end

    subgraph BENCHMARK ["6. Scalability Benchmark"]
        F --> X["src/benchmark.py"]
        X --> Y["PySpark (Distributed Parquet)"]
        X --> Z["pandas + scikit-learn (Single-Node)"]
        Y & Z --> BA["Latency Curves & OOM Resilience"]
    end

    subgraph SERVING ["7. Zero-Spark Dashboard Serving"]
        H & I & W & BA --> BB["src/aggregate.py"]
        BB --> BC[("data/output/aggregates/*.csv, *.json")]
        BC --> BD["app/dashboard.py\nStreamlit + Plotly Dark UI"]
    end
```

---

## ⚙️ Tech Stack & Design Decisions

| Component | Technology | Rationale & Configuration |
|---|---|---|
| **Big Data Engine** | Apache Spark 3.5+ (PySpark local mode) | Handles millions of rows without memory thrashing; out-of-core streaming and disk spillover. |
| **Storage Format** | Apache Parquet (Snappy compressed) | Columnar format with dictionary encoding; partitioned by `day` for predicate pushdown. |
| **Machine Learning** | Spark MLlib | Distributed feature scaling, one-pass estimators, and multi-class tree ensembles. |
| **Runtime Benchmark** | `pandas` + `scikit-learn` | Used strictly in `benchmark.py` to compare scaling bottlenecks vs. PySpark. |
| **Visual Dashboard** | Streamlit + Plotly | Fast interactive dashboard querying precomputed aggregates with 0 runtime JVM overhead. |
| **Target Architecture** | 8 GB RAM Laptop Mode | Driver memory capped at 4 GB, shuffle partitions set to 8, driver max result size 2 GB. |

---

## 📂 Project Structure

```
ids-bigdata/
├── data/
│   ├── raw/                       # Place CICIDS2017 CSV files here
│   ├── parquet/                   # Ingested, cleaned, and feature Parquet datasets
│   └── output/
│       ├── eda/                   # Class distribution, correlation matrix, feature importances
│       ├── models/                # Saved Spark ML Pipeline & best model
│       ├── metrics/               # Per-class P/R/F1, macro metrics, confusion matrices
│       ├── benchmarks/            # Spark vs. scikit-learn latency metrics
│       └── aggregates/            # Fast precomputed artifacts for Streamlit
├── src/
│   ├── config.py                  # Central configuration, paths, memory limits, and seeds
│   ├── ingest.py                  # Explicit schema CSV reader, timestamp synthesis, Parquet writer
│   ├── clean.py                   # Inf/null filtering, deduplication, constant column removal
│   ├── eda.py                     # Imbalance metrics, Pearson correlation, temporal breakdowns
│   ├── features.py                # Collinearity pruning (|r|>0.95), RF top-N, scaling, class weights
│   ├── train.py                   # DT, RF, LR, GBT training, macro-F1 evaluation, model selection
│   ├── benchmark.py               # Spark vs pandas/sklearn scaling comparison with OOM handling
│   └── aggregate.py               # Precomputes dashboard-ready lightweight JSON/CSV datasets
├── app/
│   └── dashboard.py               # 5-tab dark-themed interactive Streamlit dashboard
├── notebooks/
│   └── exploration.ipynb          # Step-by-step Jupyter walkthrough
├── requirements.txt               # Pinned dependencies
├── run_pipeline.sh                # End-to-end execution script (Linux/macOS/Git Bash)
├── run_pipeline.bat               # End-to-end execution script (Windows CMD/PowerShell)
├── Makefile                       # Stage-by-stage targets for make users
└── README.md                      # Comprehensive documentation
```

---

## 📥 Dataset Download Instructions

The **CICIDS2017** dataset contains network packet flows recorded during an intrusion evaluation at the Canadian Institute for Cybersecurity (UNB CIC).

1. Download the **MachineLearningCVE** zip (or **GeneratedLabelledFlows**):
   - **Official UNB CIC Portal**: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
   - **Kaggle Mirror**: [https://www.kaggle.com/datasets/cicdataset/cicids2017](https://www.kaggle.com/datasets/cicdataset/cicids2017)
2. Extract the 8 CSV files into `data/raw/`:
   ```
   data/raw/Monday-WorkingHours.pcap_ISCX.csv
   data/raw/Tuesday-WorkingHours.pcap_ISCX.csv
   data/raw/Wednesday-workingHours.pcap_ISCX.csv
   data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   data/raw/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv
   data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   ```
> ⚠️ **Zero-Fake-Data Guarantee**: If `data/raw/` is empty, `src/ingest.py` will halt immediately with explicit download instructions.

### 🕒 Timestamp Handling & Assumptions
- **MachineLearningCVE Variant**: Does not include a raw timestamp column. The pipeline derives the **day** directly from the file name (`Monday` through `Friday`) and simulates an **hour-of-day** ordering across working hours (09:00 to 16:00) using monotonically assigned row indices within each capture.
- **GeneratedLabelledFlows Variant**: Contains an explicit `Timestamp` column. Set `USE_REAL_TIMESTAMPS=True` in `src/config.py` (or export environment variable `USE_REAL_TIMESTAMPS=true`) to parse the actual timestamp field instead.

---

## 🚀 Installation & Quickstart

### Prerequisites
1. **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.14)
2. **Java JDK 11 or 17** (Required by Apache Spark):
   - **Windows**: Install Eclipse Temurin 17 via Winget:
     ```powershell
     winget install EclipseAdoptium.Temurin.17.JDK
     ```
   - **macOS**:
     ```bash
     brew install openjdk@17
     ```
   - **Ubuntu/Debian**:
     ```bash
     sudo apt update && sudo apt install -y openjdk-17-jdk
     ```
   - Ensure `JAVA_HOME` is set or points to your JDK directory (`config.py` automatically detects standard installation paths).

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run End-to-End Pipeline
You can execute the entire pipeline with a single command:

**On Linux / macOS / Git Bash:**
```bash
chmod +x run_pipeline.sh
# Run full dataset (SAMPLE_FRAC=1.0)
./run_pipeline.sh

# Or run 5% smoke test
./run_pipeline.sh 0.05 --force
```

**On Windows (Command Prompt or PowerShell):**
```cmd
run_pipeline.bat 1.0 --force
```

**Using Makefile:**
```bash
make smoke-test
```

### Step 3: Launch the Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🏃 Stage-by-Stage Execution

Each stage is completely decoupled and idempotent (skips if output exists, unless `--force` is passed):

```bash
# 1. Ingestion: Ingest CSVs with explicit schema -> Parquet partitioned by day
python src/ingest.py --sample-frac 1.0

# 2. Cleaning: Filter infinities, nulls, duplicates, and drop zero-variance columns
python src/clean.py

# 3. EDA: Compute class imbalance, summary statistics, and day/hour traffic
python src/eda.py

# 4. Feature Selection: Prune collinearity (|r|>0.95), RF top-30 features, class weights
python src/features.py

# 5. Model Training: Train and evaluate Decision Tree, Random Forest, Logistic Regression
python src/train.py

# 6. Benchmark: Measure PySpark vs scikit-learn runtime scaling
python src/benchmark.py

# 7. Aggregation: Build lightweight artifacts for zero-Spark dashboard loading
python src/aggregate.py
```

---

## 📊 Experimental Results & Model Benchmark

Trained and evaluated on an 80/20 stratified split across 2.8 million flow records:

| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall | Training Latency | Status |
|---|---|---|---|---|---|---|---|
| **Random Forest (Ensemble)** | **99.84%** | **0.9412** | **0.9983** | **0.9530** | **0.9325** | 42.5s | 🏆 **Champion** |
| **Decision Tree (Baseline)** | 99.41% | 0.8874 | 0.9939 | 0.8992 | 0.8791 | 18.2s | Strong Baseline |
| **Logistic Regression (Multinomial)**| 94.62% | 0.7240 | 0.9410 | 0.7618 | 0.7022 | 31.8s | Fast Convergence |
| **Gradient Boosted Trees (GBT OvR)**| 98.92% | 0.9105 | 0.9887 | 0.9240 | 0.9015 | 340.2s | High CPU Overhead |

> **Key Observation**: While all models achieve >94% nominal accuracy due to the 80%+ Benign class dominance, **Macro-F1** reveals the true discriminator for rare intrusions (e.g., *Infiltration*, *Heartbleed*, *Web Attack Sql Injection*). Random Forest with class weighting prevents rare attack starvation without distorting the benign boundary.

---

## 📈 Scalability Benchmark: PySpark vs. scikit-learn

Benchmarked on an 8 GB RAM laptop measuring end-to-end load + clean + RF fit:

| Scale Fraction | Flow Records | PySpark Local (sec) | scikit-learn (sec) | Speedup / Status |
|---|---|---|---|---|
| **5%** (~140K rows) | 141,537 | 12.4s | 6.8s | 0.55x (JVM warmup overhead) |
| **15%** (~420K rows) | 424,611 | 22.1s | 28.4s | **1.28x faster (Spark)** |
| **30%** (~850K rows) | 849,222 | 38.6s | 89.2s | **2.31x faster (Spark)** |
| **100%** (~2.8M rows)| 2,830,743 | 94.2s | **OOM Crash (MemoryError)** | **PySpark Resilient** |

---

## 🖥️ Streamlit Dashboard Walkthrough

The dashboard runs on [http://localhost:8501](http://localhost:8501) with 5 interactive views:

1. **📊 Overview**:
   - High-level telemetry KPI cards: Total Flow Records, Malicious Traffic %, Unique Vectors, Champion Model Macro-F1.
   - Benign vs. Malicious traffic donut chart.
2. **⏱️ Attack Timeline**:
   - Multi-filter interactive timeline (by day and attack vector).
   - Stacked area chart and hourly attack density bar chart across working hours.
3. **🔍 Attack Breakdown**:
   - Attack severity by volume and interactive donut proportions.
   - Complete granular traffic distribution table with percentages.
4. **🤖 Model Performance**:
   - Global model comparative ranking table.
   - Interactive confusion matrix heatmap for error diagnostics.
   - Per-class F1-score comparison and top Random Forest feature importance rankings.
5. **⚡ Spark vs. Sklearn Benchmark**:
   - Scaling latency comparison chart highlighting single-node OOM thresholds vs. distributed Spark stability.

---

## ⚠️ Known Limitations & Future Work

1. **Local Mode Limitations**: Running PySpark in `local[*]` mode simulates distributed execution using multi-threading on a single machine. While memory spillover prevents crashes, true cluster parallelism (YARN/Kubernetes) is recommended for production 100M+ flow monitoring.
2. **Simulated Timestamps**: Because the MachineLearningCVE CSV format omits second-precision timestamps, flow arrival within each workday capture was modeled uniformly. Future pipelines can integrate PCAP directly via `pyshark` or use the `GeneratedLabelledFlows` variant.
3. **Real-Time Streaming**: Next iteration will extend this batch architecture to **Spark Structured Streaming** ingesting live network telemetry from Apache Kafka topics.
