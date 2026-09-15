# PFAS adsorbent recommender

Interactive Streamlit interface that ranks commercial adsorbents for one or more
PFAS under user-specified water chemistry. Comparable database records are used
first; a pre-trained XGBoost ensemble predicts log K<sub>d</sub> at the 25 mg/L
reference dose where no comparable record exists.

**Try it online:** https://pfas-adsorbent-recommender.streamlit.app

## Run locally

```bash
pip install -r requirements.txt
streamlit run scripts/adsorbent_recommender/app.py
```

Run the command from the repository root so `.streamlit/config.toml` (the theme)
is applied.

## Contents

| Path | What it is |
| --- | --- |
| `scripts/adsorbent_recommender/` | Streamlit app, recommendation service and bundle export |
| `scripts/ML_model_training/backend/` | Feature construction used by the saved model pipeline |
| `scripts/DB/` | Normalization rules and mapping tables for user inputs |
| `output/ML_logKd/Global_xgb_combination_42_20260907_141359/` | Deployed model bundle: ensemble, feature support, adsorbent catalog snapshot, PFAS feature cache and reference-dose database evidence |

A model bundle is self-contained: everything the app reads at run time is inside
its directory. The app loads the most recent bundle under `output/ML_logKd/`;
set `ADSORBENT_BUNDLE_DIR` to use a different one.
