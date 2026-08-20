### 🎯 SentinelFlow v2: Complete Production Prototype Roadmap
**Operational Systems Design, Low-Latency Orchestration, & On-Demand Model Retraining**
*Target Submission Date: August 17, 2026*

---

### 1. Architectural Blueprint: Integrated Multi-Source Orchestration

The SentinelFlow v2 prototype transitions from a static offline research pipeline into a dynamic, multi-source orchestration system. It is designed to handle heterogeneous real-time transactional streams, ingest external regulatory warnings, and run on-demand model retraining directly from a visual control console.

```
                              [ STREAMLIT ORCHESTRATION CONSOLE ]
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
         [ Live SecOps Dashboard ]                         [ Model Orchestrator Panel ]
           - Real-time Transaction Stream                    - Feature Toggle Checklist
           - Calibrated Risk Scoring & Holds                 - Retrain Trigger Button
           - TreeSHAP Reason Codes Display                   - Model Arena & Selection
                       │                                               │
                       ▼                                               ▼
         [ MULTI-SOURCE INGESTION GATEWAY ] ◄─────────────────[ FAST-API ENGINE ] 
                       │                                      (Registry / /retrain / /score)
         ┌─────────────┼─────────────┐                                 │
         ▼             ▼             ▼                                 ▼
   [ Core Stream ]  [ Profiles ]  [ Govt Feeds ]            [ MODEL REGISTRY STORAGE ]
     (Transaction)   (Redis Cache)  (Blacklist)               - M1: Anchor-Free Robust (Default)
         │             │             │                        - M2: Chronological Baseline
         └─────────────┼─────────────┘                        - M3: Audited Leaky (Demo)
                       ▼                                      - M4: Custom User-Retrained
                [ DECISION CORE ]                                      
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
   [ Deterministic ]           [ Calibrated ML ]
   (Blacklist Match)           (GBDT Ensemble)
         │                           │
         └─────────────┬─────────────┘
                       ▼
         [ COST-SENSITIVE THRESHOLD ]
                       │
                       ▼
         [ PREVENTION ENGINE ] ──► [ Escrow Hold & Outbound Webhooks ]
```

---

### 2. High-Level Phase 2 Execution Plan & Milestones

#### 📅 Weeks 1–2: Ingestion Protocols & Data Schema Engineering
*   **Objective:** Define and implement robust, structured data schemas to ingest core transactions, cross-channel bank profiles, and external real-time regulatory feeds.
*   **The Ingestion Gateway:**
    *   **Core Transaction Schema:** A structured JSON payload representing incoming customer transactions (transfer amount, transaction type, channel ID, recipient hashes).
    *   **Cross-Channel Profile Schema:** An in-memory cached state representing the customer's near-real-time behavior across multiple banking channels (Debit Card, Credit Card, UPI, Net Banking) to detect coordinated velocity anomalies.
    *   **Government Cyber Fraud Alert Feed:** A simulated real-time API endpoint representing national cybercrime warning tickets (suspicious IP subnets, flagged device IDs, blacklisted account numbers).
*   **On-Demand Orchestration Schema:** 
    *   Define the payload schema for the `/retrain` API endpoint. This schema allows the Streamlit dashboard to send configuration dictionaries specifying active feature lists, oversampling ratios, and hyperparameter overrides.

#### 📅 Weeks 3–4: Core FastAPI Backend & Model Registry Development
*   **Objective:** Code the operational backend containing the feature reconstruction engines, the model registry, and the on-demand retraining pipeline.
*   **The Feature Reconstruction Engine:**
    *   **Behavioral Moments:** Calculate 14 statistical indicators (row-level means, std, zero-rates, IQR) dynamically on incoming data.
    *   **Cyclical Temporal Parsing:** Isolate the account opening date (`F3888`), calculate cyclical month and day-of-week sine/cosine coordinates, and completely omit absolute `elapsed_days` to prevent cohort timeline drift.
    *   **Non-Destructive Categorical Encoding:** Implement an in-memory, fold-safe categorical encoding track. Map low-cardinality features ordinally, and apply Laplace-smoothed target encoding to high-cardinality features.
    *   **Isolation Forest Anomaly Score:** Run a pre-trained, fold-safe Isolation Forest to output a single continuous outlier metric.
*   **The Model Registry Module:**
    *   Implement a model registry within the FastAPI filesystem storing **four distinct, serialized model versions** (each occupying less than 5 MB of memory):
        1.  **Model 1 (Anchor-Free Robust - Default):** The mathematically clean GBDT ensemble utilizing our 105 non-leaky behavioral and cyclical temporal features.
        2.  **Model 2 (Chronological Baseline):** The baseline GBDT model that retains the absolute `elapsed_days` timeline anchor, serving as an active comparison point.
        3.  **Model 3 (Audited Leaky):** The baseline model with the six post-hoc administrative leakage features (`F3912`, `F2230`, `F3886`, `F3889`, `F3891`, `F3892`) deliberately left active to demonstrate the impact of leakage to the judges.
        4.  **Model 4 (User-Custom Retrained):** The dynamic model slot occupied by the model retrained on-the-fly via the dashboard.
*   **The On-Demand Retraining Pipeline:**
    *   Write a `/retrain` endpoint. When triggered, the API loads the raw dataset, excludes any features deselected by the user, runs the complete preprocessing and SMOTE balancing loop, fits a new XGBoost + LightGBM ensemble, applies **PCHIP-smoothed Isotonic Calibration**, optimizes the cost-sensitive threshold, and registers the output as Model 4.

#### 📅 Week 5: Streamlit Security Operations & Orchestration Console
*   **Objective:** Build an interactive, professional frontend dashboard that acts as the visual showcase for your live hackathon demonstration.
*   **The Live SecOps Dashboard Panel:**
    *   **Transaction Simulator:** Allows judges to select sample accounts, adjust transaction variables, and submit live scores through the API.
    *   **Real-Time Alert Queue:** A prioritized list of high-risk transactions. High-risk entries display a colored alert status with their calibrated probability.
    *   **The Explainability Console:** Clicking an alert renders:
        *   The continuous **Calibrated Predictive Risk Score**.
        *   An interactive **Local Network Ego-Graph Visualizer** showing how funds are flowing from the suspect account to immediate neighbors.
        *   A horizontal bar chart of **Taylor-scaled TreeSHAP attribution scores** explaining the exact reason codes behind the alert.
*   **The Model Orchestration & Retraining Panel:**
    *   **The Feature Control Board:** A checklist of all raw features. Judges can toggle individual features (such as adding back a leaky feature like `F2230` or `F3891`) to test model responses.
    *   **The Model Arena:** A dropdown menu allowing users to swap the active API inference model between the four stored registry models in real-time.
    *   **The Retrain Trigger:** A button that sends the active checklist configuration to the `/retrain` endpoint, displays a real-time progress spinner as the backend re-runs the entire pipeline, and instantly updates the validation metrics (PR-AUC, Macro F1, Total Cost) on the screen.

#### 📅 Week 6: Production Hardening, Retraining Guardrails, & Technical Report
*   **Objective:** Configure the final containerized architecture, enforce retraining safeguards, and finalize the LaTeX technical report.
*   **Production Caching & Retraining Guardrails:**
    *   **Write-Back Caching & Persistence:** Configure the backend to write transaction logs to a Redis in-memory cache first to maintain sub-100ms latency, backed up by **Redis Append-Only File (AOF) persistence** set to `appendfsync-every-second` to prevent data loss on node crashes.
    *   **Redis TTL Sliding Windows:** Apply explicit Time-To-Live (TTL) keys (e.g., 24-hour and 7-day) to transactional edges in the Redis ego-graph, naturally evicting stale data.
    *   **Retraining Label-Delay Buffer:** Implement a strict **90-day lookback buffer** in your retraining data pipeline. Retraining operations only ingest transaction cohorts older than 90 days, ensuring that accounts with unresolved, lagging fraud disputes are excluded from training to prevent learning from biased labels.
    *   **Monotonicity Assertion Check:** During automated retraining, the pipeline runs an assertion verifying that the first derivative of the PCHIP-smoothed Isotonic spline is strictly positive ($f'(x) > 0$) across the $[0, 1]$ range. If the spline fails this check, retraining is aborted, preventing inverted SHAP attributions.
*   **Technical Report Finalization:**
    *   Compile your final LaTeX report, replacing all placeholders with the **actual empirical latency metrics, pipeline throughput, and audited cost-benefit statistics** gathered from your running Phase 2 prototype.

---

### 4. Operational Checkpoints & Validation Rubric

Before submitting on **August 17, 2026**, your prototype must pass three strict operational checks:

1.  **The API Latency SLA:**
    *   *Check:* Submit a payload to the FastAPI `/score` endpoint.
    *   *SLA:* The combined latency (feature extraction + ego-graph calculation + ensemble scoring + PCHIP calibration + cost-sensitive evaluation) must remain strictly **below 100ms** (target: <50ms).
2.  **The Registry Hot-Swap Test:**
    *   *Check:* Swap the active model on the Streamlit dashboard between Model 1 (Anchor-Free) and Model 3 (Leaky).
    *   *SLA:* The API gateway must hot-swap the active inference weights instantly, recording a **0ms downtime** during the transition.
3.  **The Retraining Integrity Check:**
    *   *Check:* Trigger an on-demand retrain with a custom feature checklist.
    *   *SLA:* The pipeline must run completely end-to-end, calibrate probabilities using a strictly monotonic PCHIP curve, and register Model 4 in under **15 seconds** (achievable due to the compact dataset size and fast GBDT training speeds).

---

### Phase 2 Execution Protocol

This roadmap represents the final, unifying blueprint for your Phase 2 hackathon submission. It balances advanced, mathematically sound data science with highly robust systems engineering and a stunning visual demonstration interface designed to impress the panel of judges.
