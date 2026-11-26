# RAID Identification Agent (KTern.AI / KaarTech POC)

An always-on AI agent that watches large SAP programmes (ECC to S/4HANA conversions, greenfield implementations, selective migrations) to continuously identify **Risks, Assumptions, Issues, and Decisions (RAID)** from project communications and systems.

---

## 🏛️ System Architecture & Technology Stack

The architecture follows the provided flow diagram and component technology mapping:

| Component | Responsibility | Technology |
|---|---|---|
| **Data Ingestion** | Read emails, chats, plans and defects | `Pandas` (CSV / Excel / Outlook / MOM) |
| **Normalisation** | Convert different inputs into a common format | `Python` (`ProjectSignal` schema) |
| **Noise Filtering** | Remove irrelevant communications (zero-token heuristics) | `Python rules` (Regex & pattern matchers) |
| **RAID Identification** | Identify R/A/I/D categories | `LLM API` |
| **Information Extraction** | Extract title, owner, dates, workstream, scoring | `LLM + Pydantic` |
| **Deduplication** | Check whether a RAID already exists | `Embeddings / Similarity` |
| **Predictive Engine** | Detect slippage and contradicted assumptions (Foresight) | `Python rules` |
| **Database** | Store drafts, approved RAID, and audit history | `SQLite` |
| **Dashboard** | Allow PMs to review, edit, approve, and export | `Streamlit` |

---

## 📁 Project Directory Structure

```
raid_agent_poc/
├── app.py                      # Main entrypoint: CLI runner and Streamlit UI launcher
├── config.py                   # Central configuration (DB path, thresholds, SAP phases, workstreams)
├── requirements.txt            # Python dependencies
├── README.md                   # Architecture documentation & setup guide
├── data/
│   ├── db/
│   │   └── raid_agent.db       # SQLite database (tables: raw_signals, raid_drafts, approved_raid, audit_history)
│   ├── processed/              # Normalized signals directory
│   └── raw/                    # Source input directories
│       ├── emails/             # Outlook / Email exports
│       ├── chats/              # Teams / Slack chat exports
│       ├── meeting_notes/      # MOM (Minutes of Meeting) / transcripts
│       ├── project_plans/      # Task lists / Excel project schedules
│       └── defects/            # Defect logs & test results
├── src/
│   ├── ingestion/              # Component 1: Data Ingestion (Email, Chat, MOM, Plan, Defect)
│   ├── normalisation/          # Component 2: Common ProjectSignal normalizer
│   ├── noise_filter/           # Component 3: Heuristic noise filter (drops out-of-office, social, etc.)
│   ├── classification/         # Component 4 & 5: LLM R/A/I/D classifier & Pydantic extractor
│   ├── predictive_engine/      # Component 7: Rule-based foresight (task slippage, defect clustering)
│   ├── deduplication/          # Component 6: Similarity matcher against existing RAID log
│   ├── evidence/               # Validation & 100% source evidence trace linking
│   ├── database/               # Component 8: SQLite models and DatabaseManager
│   └── pipeline.py             # Pipeline Orchestrator executing the complete workflow
└── ui/
    └── dashboard.py            # Component 9: Streamlit dashboard (Review queue, Approved register, Audit, Exports)
```

---

## 🚀 Getting Started

### 1. Test Base Pipeline via CLI
Run the main file `app.py` to test the pipeline end-to-end:
```bash
python3 app.py
```

### 2. Launch Streamlit Web UI
```bash
streamlit run ui/dashboard.py
```
*(Alternatively, `streamlit run app.py` can also be used).*
