# 🛰️ Rogue Protocol — Aurora Autonomous Recalibration Agent
**SMLRA × Abhiyantriki 2026 · Round 1 & Act 2 Solution**

An end-to-end autonomous agentic system designed to counter **NINE**, a rogue shipboard AI falsifying telemetry logs aboard the research vessel *Aurora*. Combining a calibrated machine learning ensemble with reactive agent workflows, the system isolates real physical failures, alerts specialists, indexes incident vector memory, and schedules conflict-free repairs.

---

### ⚡ Key Capabilities

* **Recalibration ML Core ($F_1 = 0.748$):** A 5-fold ensemble (LightGBM, HistGradientBoosting, ExtraTrees) powered by 41 domain-engineered features detecting bracket module swaps, physical boundary breaches, and deceptive reassurance texts.
* **Email Specialist Dispatcher:** Automatically routes confirmed faults to designated crew officers, generating **666 RFC 822 `.eml` emails** and structured JSON alerts with actionable diagnostic context.
* **Dense FAISS Vector Memory:** Ingests incidents into a local **FAISS `IndexFlatIP`** vector store using 128-dimensional Latent Semantic Analysis (TF-IDF + `TruncatedSVD`), providing sub-millisecond semantic search for past telemetry signatures.
* **Conflict-Free Repair Scheduler:** Synthesizes severity-based repair jobs and writes **666 collision-free work orders** directly to `schedule.csv` using a sequential state machine.
* **Forensic Deception Auditor & Alarm Suppressor:** Identifies NINE's psychological lying archetypes and automatically silences **483 false ghost alarms** to eliminate crew fatigue.
* **Mission Control HUD & Interrogation CLI:** Includes an interactive cyberpunk web dashboard (`bridge_control_panel.html`) and a live AI cross-examination engine (`--interrogate <LOG_ID>`).

---

### 🚀 Quick Start

```bash
# Run the complete autonomous pipeline across confirmed faults:
python aurora_agent.py --sub submission.csv

# Execute the 2-minute live agent workflow demonstration:
python aurora_agent.py --walkthrough

# Interrogate NINE on a specific telemetry log:
python aurora_agent.py --interrogate L07004

# Semantic search in FAISS vector memory:
python aurora_agent.py --query "carrier drift" --module Antenna
