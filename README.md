# PFAS adsorbent recommender

Code for *From Literature to Adsorbent Recommendation: LLM-Assisted Data Curation
and Machine Learning Modeling of PFAS Adsorption* (Yan and Helbling). The
repository contains:

1. **The adsorbent recommender.** An interactive Streamlit interface that ranks
   commercial adsorbents for one or more PFAS under user-specified water
   chemistry. It uses comparable database records first. Where none exist, a
   pre-trained XGBoost ensemble predicts log K<sub>d</sub> at the 25 mg/L
   reference dose.
2. **The literature-to-model workflow behind it.** Document preprocessing,
   LLM-based data extraction and its evaluation, database merging and
   harmonization, PFAS descriptor generation, and the logK<sub>d</sub>
   machine-learning analyses reported in the manuscript.

**Try the recommender online:** https://pfas-adsorbent-recommender.streamlit.app

## Run the recommender locally

```bash
pip install -r requirements.txt
streamlit run scripts/adsorbent_recommender/app.py
```

Run the command from the repository root so `.streamlit/config.toml` (the theme)
is applied. A model bundle is self-contained: everything the app reads at run
time is inside its directory. The app loads the most recent bundle under
`output/ML_logKd/`; set `ADSORBENT_BUNDLE_DIR` to use a different one.

## Investigating PFAS outside the database

PFAS concentrations can be entered in ng/L, µg/L, or mg/L. Use **Import a PFAS
mixture from CSV** to upload a quantified mixture, preview all rows, and append
it after validation. **Download example CSV** provides the expected columns
and a working example. The model continues to use mg/L internally; changing
a displayed unit preserves the concentration. See
`scripts/adsorbent_recommender/README.md` for the CSV format and validation rules.

Search the reviewed PFAS list as usual. For an identifier outside that list,
choose whether it is a name, CAS number, or SMILES. Names and CAS numbers can
be resolved through PubChem; select the intended result, inspect its structure,
and choose **Confirm structure and add PFAS**. Direct SMILES input is always
available, including when PubChem cannot find a compound or is unavailable.
Supply one defined PFAS parent molecule, without salt counterions, mixture
components, wildcard atoms, or undefined polymer repeat units.

The existing XGBoost ensemble can use RDKit features computed from that
structure without running OPERA or adding it to the reviewed workbook. The
**PFAS support** panel distinguishes reviewed versus runtime features from
whether the compound was actually present in model training. Both cached and
newly computed unseen PFAS receive exploratory-prediction warnings. Descriptor
range warnings and the three nearest training PFAS provide additional context;
structural similarity and ensemble SD are not calibrated confidence estimates.
Mixture scores still evaluate every entered PFAS and retain the maximin rule.

The bundle remains version 4; result downloads use schema version 5. Older
bundles retain cached screening. Repackage an existing bundle without refitting:

```bash
python -m adsorbent_recommender.bundle <bundle-directory>  # run with PYTHONPATH=scripts
```

Repackaging links actual fitted records to curated structures, checks descriptor
parity, and records the descriptor/RDKit versions. Unseen-PFAS generation is
enabled only when the selected PFAS inputs are computable and parity passes;
the runtime RDKit version must match the packaged check.

## Interpreting recommendations

The app is for preliminary adsorbent screening. It compares catalog materials
under the entered conditions at a common 25 mg/L dose; a high ranking is not
evidence that a treatment system will meet a site's removal target. Validate
shortlisted materials using the relevant water and operating conditions before
selection.

The **Basis** column distinguishes comparable database records from XGBoost
estimates. The **Caveats** column points to limitations for each ranked score;
open **Candidate details** to read them. Model predictions are less reliable for
unseen PFAS, adsorbents and study conditions or at very low and high log Kd. The
ensemble standard deviation is model disagreement, not a calibrated prediction
interval. Mixture rankings evaluate PFAS separately and do not account for
competitive adsorption. The app does not predict breakthrough, media lifetime,
removal targets, or cost.

The study manuscript has not yet been submitted, so the app does not display a
paper link. When a public manuscript or publication URL is available, set
`ADSORBENT_PAPER_URL` to its full HTTPS URL; the app will show a **Read the study**
link in its scope and limitations panel.

## Repository contents

| Path | What it is |
| --- | --- |
| `scripts/adsorbent_recommender/` | Streamlit app, recommendation service and bundle export |
| `output/ML_logKd/Global_xgb_combination_42_20260907_141359/` | Deployed model bundle: ensemble, feature support, adsorbent catalog snapshot, PFAS feature cache and reference-dose database evidence |
| `scripts/Preprocessing/` | Docling conversion, rule-based repair of conversion errors, section/figure/table handling, chunking |
| `scripts/Data_Extraction/` | Abstract screening, classification, enumeration, extraction, evaluation, and annotation apps |
| `chains/`, `schemas.py` | LangChain task chains, LLM provider configuration, and output schemas |
| `prompts/` | Jinja2/YAML prompts, label definitions, and few-shot examples for every LLM step |
| `scripts/DB/` | Merging, unit and vocabulary normalization, endpoint→K<sub>d</sub> conversion, QC flags, data audit |
| `scripts/PFAS/` | PFAS descriptor generation from SMILES (RDKit, with optional OPERA and PFAS-Atlas) |
| `scripts/ML_model_training/` | logK<sub>d</sub> modeling: training, comparisons, ablations, split strategies, SHAP |

Each workflow folder has its own `README.md` with detailed run instructions.

## Reproducing the manuscript analyses

### Environments

The analyses used two Python environments:

```bash
pip install -r requirements-extraction.txt   # Preprocessing, Data_Extraction (Python 3.11)
pip install -r requirements-modeling.txt     # DB, PFAS, ML_model_training
```

The LLM steps need API keys. Copy `.env.example` to `.env` at the repository
root and fill in `OPENAI_API_KEY` (GPT-4.1, GPT-5.5) and/or `TOGETHER_API_KEY`
(open-weight models via Together AI). Never commit `.env`.

### Data

This repository contains code only. The source articles are copyrighted and
are not redistributed. The curated database, human annotations, and LLM
outputs are available as described in the manuscript's Data availability
statement. By default, the scripts expect the following layout, with the data
folders placed next to this repository:

```
<parent>/
├── pfas-adsorbent-recommender/      # this repository
├── Adsorbent_Data/<study_id>/       # retrieved main_paper / supplementary_material files
├── Processed/                       # Docling JSON, JSON-first records, chunks
└── web of science search/           # Web of Science export and within_scope_records.xlsx
```

Most locations can be overridden with command-line options or environment
variables, for example `ADSORBENT_SOURCE_STUDIES_DIR`,
`ADSORBENT_PROCESSED_DIR`, `ADSORBENT_STUDY_RECORDS_XLSX`, `PFAS_PROPS_PATH`,
and `PFAS_DATA_DIR`. See the folder READMEs and each script's `--help`.
Studies are assigned to the development ("Training"), validation, and
application sets in `scripts/*/workflow_config.py`, where `ACTIVE_STUDIES`
selects the batch to run.

### Pipeline order and manuscript mapping

| Step | Script(s) | Manuscript |
| --- | --- | --- |
| 1. Abstract screening (GPT-4.1) | `scripts/Data_Extraction/screen.py` | Methods: Data preparation |
| 2. PDF/DOCX → Docling JSON | `scripts/Preprocessing/convert_to_docling_json.py` | Methods: Data preparation |
| 3. Repair conversion errors; keep relevant sections; separate figures/tables | `scripts/Preprocessing/docling_json_preprocess.py` (+ `docling_text_repair_decisions.csv`) | Text S1 |
| 4. Chunking (HybridChunker, 512 tokens) | `scripts/Preprocessing/chunk_json_input.py` | Methods: Data preparation |
| 5. Classification | `scripts/Data_Extraction/classify.py` | Fig. 2c–e; Methods: Data extraction |
| 6. Enumeration (adsorbent and water-type lexicons, performance identifiers) | `scripts/Data_Extraction/enumerate.py` | Fig. 2f |
| 7. Extraction (adsorbent properties, performance, conditions, unit recovery) | `scripts/Data_Extraction/extract.py` | Fig. 2f–g; Table S3 |
| 8. Evaluation against human annotations | `scripts/Data_Extraction/evaluation_{classification,enumeration,extraction}.py`; annotation apps `*_UI.py` | Fig. 2c, 2f, 2g |
| 9. PFAS descriptors | `scripts/PFAS/generate_pfas_features.py` | Methods: Data merging and harmonization |
| 10. Merge, harmonize, convert endpoints to K<sub>d</sub>, QC flags | `scripts/DB/build_database.py` (runs `merge.py`, `normalize.py`, `data_audit.py`) | Fig. 3; Text S2 |
| 11. Algorithm comparison | `scripts/ML_model_training/run_logkd_algorithm_comparison.py` | Text S4, Fig. S2 |
| 12. Missing-data strategy | `run_logkd_missing_strategy_comparison.py` | Text S5, Fig. S3 |
| 13. Global vs. adsorbent-specific models | `run_logkd_model_family_comparison.py` | Fig. 4a |
| 14. Held-out SHAP across outer repeats | `logkd_shap_outer_robustness.py` | Fig. 4b, 4d; Fig. S6 |
| 15. Feature-family ablation | `run_logkd_feature_family_only.py` | Fig. 4c |
| 16. Dosage / C<sub>0</sub> ablation | `run_logkd_dosage_c0_ablation.py` | Fig. 4e; Fig. S7 |
| 17. Data-splitting strategies | `run_logkd_split_strategy_comparison.py` | Fig. 5b–d; Fig. S9 |
| 18. K<sub>d</sub> vs. logK<sub>d</sub> target | `run_kd_vs_logkd_random_row_comparison.py` | Methods: ML model development |
| 19. Final deployable ensemble | `build_final_logkd_model.py` | Methods: Adsorbent recommender |

Supporting modeling scripts: `train_logkd_model.py` runs a single training
and evaluation. `run_logkd_feature_threshold_comparison.py` and
`run_allocation.py` compare the feature-screening and inner-fold allocation
choices. `plot_logkd_results.py` and `logkd_residual_review.py` review
results. `prompts/extract_template_without_enumeration.j2` is the single-step
extraction prompt used only for the with/without-enumeration comparison.

### Tests

```bash
cd scripts/Preprocessing && python -m pytest tests
cd scripts/ML_model_training && python -m pytest tests
```

Tests that need the study data folders are skipped when those folders are
absent.

## License

Released under the MIT License (see `LICENSE`). If you use this code, please
cite the accompanying article (see `CITATION.cff`).
