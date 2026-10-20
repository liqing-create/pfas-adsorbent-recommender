# PFAS adsorbent recommender

## Assumptions and interpretation

The app shortlists catalog materials for experimental testing. It compares
adsorption affinity; it does not establish site treatment performance, service
life, breakthrough, regulatory compliance, or commercial suitability.

- **Initial PFAS concentration:** enter a positive quantified value for each
  compound. Manual entry starts blank and clears after adding each PFAS; the
  chosen unit is retained for the next entry. ng/L, µg/L, and mg/L are supported.
  Changing units preserves the physical concentration. There is no assumed
  PFAS concentration; example-file values are illustrative.
- **Water type:** unspecified water is treated as ultrapure water when no
  background additives are reported. Reported inorganic or organic additions
  promote unspecified/ultrapure water to synthetic water under the database
  convention. Other explicitly selected matrices are retained.
- **Temperature:** a blank uses the selected bundle's fixed default, currently
  25 °C, in both model predictions and evidence matching. It is an assumed
  screening condition, not a reported measurement. Results label it as a default;
  downloaded run workbooks preserve the submitted blank.
- **Other unreported chemistry:** blank pH, organic carbon, TDS, and ionic
  strength are not assigned measured zero values. Numeric matching requirements
  apply only where values are supplied or derived. Listed inorganic additives
  can supply ionic strength and ion indicators; an explicit ionic-strength
  measurement takes precedence. Unchecked presence controls impose no positive
  presence requirement; they do not prove a constituent is absent. Missing
  features follow the existing training and normalization conventions.
- **Common adsorbent dose:** the current bundle compares candidates at
  25 mg/L of adsorbent. This is distinct from the initial PFAS concentration.
  Comparable database values use exact isotherm recalculation where possible,
  otherwise observations in the 20–30 mg/L near-dose window. Model fallbacks
  use the exact reference dose. Other fixed conditions follow the selected
  bundle and are preserved in the result's screening conditions.
- **Evidence first:** comparable records require verified PFAS and product
  identity, standardized water type, and applicable numeric tolerances. A blank
  temperature still imposes the default temperature requirement (±2 °C in the
  current bundle). Unreported record temperatures do not satisfy it. Related
  records with different or missing conditions provide context only; they do
  not contribute ranking scores. The model fills admissible unmatched pairs.
- **Mixtures:** each PFAS is evaluated independently at its own concentration
  under shared water conditions. Adsorbents are ranked by their lowest logKd
  across all entered PFAS; the corresponding compound is limiting. The mean is
  supplementary. Every PFAS must have an admissible score. Competitive
  adsorption among mixture components is not represented.
- **Scores and uncertainty:** logKd is log10 of Kd in L/g; higher indicates
  stronger affinity in this screening. The “Top-performing tier” consists of scores
  within the cross-validated error of a predicted difference between two
  adsorbents of the best score in the current class view (approximately 0.51
  log units for the current model; computed by
  `scripts/recommender_validation/cv_difference_error.py` and stored in the
  bundle as `leading_group_margin.json`). This is a screening
  heuristic, not a calibrated confidence interval, significance test, or proof
  of equivalent performance. Decimal precision does not imply predictive
  precision. Ensemble SD measures model disagreement, not calibrated uncertainty.
- **Support and caveats:** only products in the training data are ranked. The
  single **Warnings** column shows concise labels
  for recorded critical and contextual caveats. Soft rose **!** labels are critical;
  muted teal **i** labels provide supporting context. Cells without recorded
  caveats are left blank. Hovering a label shows its meaning. Select a row for
  the full warning text, affected PFAS, study concentration, and supporting records.
- **Saving:** **Keep for comparison** keeps a completed run for the current
  browser session. **Download run (.xlsx)** preserves reusable settings,
  rankings, per-PFAS evidence, and caveats together in one file.

The interface keeps these detailed assumptions here rather than adding an
assumptions panel to the screening form. Effective defaults remain identified
in existing field help and result summaries.

## Product names and disclaimer

Product names are trademarks of their respective owners and are used only to
identify materials reported in the scientific literature. Their inclusion does
not imply endorsement, recommendation, or affiliation by the authors or their
institution. Scores are research estimates of adsorption affinity derived from
published data and an XGBoost machine-learning model; they are not performance
claims or guarantees for any product. Commercial formulations may change over time and
vary between lots. This tool is provided "as is", without warranty of any kind.

The app shows this notice at the top of **How to read these results**
(`PRODUCT_NAME_DISCLAIMER` in `app.py`); keep the two texts in sync when either
is edited.

## Using the screening interface

The PFAS list holds only the PFAS represented in the model's training records
(`pfas_training_reference.csv`, 128 structures for the deployed bundle), so
screening beyond the training PFAS is possible but never offered as a default.
Search results show abbreviation, full name, and CAS together; typing any of
these searches the same reviewed compound. The bundled feature cache holds the
same 128 training PFAS and nothing else, although the shared workbook it is cut
from also lists database PFAS without eligible records and degradation products.
Another PFAS is entered by typing its name, CAS number, abbreviation, or SMILES
and choosing the Add option; it goes through the explicit structure-review path,
and its descriptors are computed from the confirmed structure. Enter its
initial concentration, choose a unit,
and add it to the mixture. Concentration and unit controls wrap as a pair of
readable fields on smaller screens. Each added entry shows its name and CAS and
a Remove action; an entry outside the training PFAS is tagged
**Not in training data**.

The interface keeps instructions to what a control does not already make
obvious: inputs carry no explanatory subtitles, and the full assumptions live
here rather than on screen. **How to read these results**, beside the results,
gives four short points (the score, ties, where the model is less reliable, and
what it does not model) and links to this README.

Screening is enabled only when all PFAS entries and shared water fields are
valid. Readiness feedback identifies what needs attention. Edits retain the
previous result with a stale-results notice; screening updates it explicitly.
After completion, **View results** jumps to the stable results heading.

The default shortlist encloses the top-performing tier separately from the
remaining candidates while keeping candidates ordered by the displayed logKd
score. Row fill identifies the broad material family (AC, resin, or CDP).
**Adsorbent studies** and **Adsorbent records** count the training studies and
records for the adsorbent with any PFAS, not for the requested combination; they
are context for choosing among candidates, not reliability grades (see below).
Scores, evidence basis,
and severity-colored warning labels remain visible; mixtures also show their
limiting PFAS. Select a row to open its evidence in a dialog over the table, so
it is visible without scrolling past the ranking;
the results have no tabs. The table header
keys class, and each warning label shows its meaning on hover in
place of a legend. Labels include **One study** (the
database score comes from a single study), **Near dose** (database observations
at 20–30 mg/L rather than exactly 25 mg/L), and **Conflicting data** (comparable
records span widely different logKd values). Other labels are explained when
they occur. Identical warnings across mixture components share one label;
the selected row's evidence preserves every explanation and its PFAS scope. The column
reserves width for all labels rather than replacing them with an unexplained
count. Complex warning sets can make the table horizontally scrollable.

Label fills are translucent so they blend with the app's light and dark themes;
the same palette is used in the table and the full explanations. Empty cells
do not produce a badge or an extra screening-tool reminder; the general scope
and assumptions remain documented here. Colors reflect the recorded caveat severity, not a confidence grade or a new
reliability score. There is no invented intermediate severity. Unknown caveat
codes retain their supplied label and text; unknown severity is marked for
review. All calculations, warning rules, source severities, and complete workbook
warning text remain unchanged.

There is no separate support or scoring panel. How scores are obtained (database
record or model, at the common 25 mg/L dose) is in **How to read these results**.
A PFAS in the training data gets no support note, since the dropdown offers only
those. When an entered PFAS is outside the training data, one warning above the ranking says its
scores are exploratory and names its three structurally closest training PFAS
with their Tanimoto similarity. **Download run (.xlsx)** includes the ranking,
all adsorbent–PFAS records, technical fields, caveats, and reloadable inputs.
When inputs have been edited, the download continues to describe the older
displayed result rather than the changed editor fields.

## Screening input validation and temperature

Leaving temperature blank uses the bundle's fixed default (25 °C in the current
bundle) for both model predictions and comparable database evidence. The
existing ±2 °C evidence tolerance applies to this effective temperature;
unreported temperatures cannot qualify. The results identify the default,
while saved inputs retain a blank temperature so it remains distinguishable
from an explicitly supplied measurement.

pH, temperature, organic carbon, TDS, and ionic strength must be finite numeric
values or blank. Organic carbon, TDS, and ionic strength cannot be negative;
reported zero values remain valid. Invalid nonblank values block screening,
preserve the entered text, and mark any displayed result as stale.
Finite pH and temperature values outside training ranges retain the existing
support warnings. The service and workbook import apply the same numeric rules.

Workbook imports validate the input table before changing the editor or saved-run
library. Formula cells, malformed files, and unsupported columns are rejected
without replacing current inputs. An invalid result already in the session
library remains removable.
The implementation fingerprint includes the shared input rules, and previous
displayed results require a new screening after the behavior update. Model
weights, data snapshots, file schema versions, and PFAS limits are unchanged.

## Adsorbent training support

Property eligibility remains separate from fitted evidence support. ML ranking
requires a verified product identity in the fitted records: the model is used
for untested pairings of PFAS and adsorbents that are each in the training data
(the combination-held-out setting), so a product with no training record is not
ranked. Every ranked product is shown with its number of DOI-identified studies
and distinct source records in the training data, with no support classification
or caveat attached. Tier placement was about as reliable for products with few
studies or records as for well-represented ones (`by_support` in
`output/Recommender_validation/tier_hidden_pairing_summary.json`, written by
`scripts/recommender_validation/tier_hidden_pairing_check.py`), because
the model predicts from adsorbent properties rather than product identity; the
counts help users weigh how widely a product has been tested, not how far to
trust its score. The service still computes the former moderate/thin policy
internally, but `scenarios.public_result` removes it, along with screened-out
products, from the page and every export. One publication's study aliases count once by normalized
DOI; distinct publications can still share experiments or systematic biases.
Source records use `extraction_dataset/source_row_index`, so isotherm-expanded
siblings do not inflate support.

The app no longer offers ML scores for products without a training record. The
service keeps the `adsorbent.include_exploratory` request field for analyses;
when set, such products are scored with a critical "Not in training" caveat.
Saved scenarios and inputs CSVs keep the `include_exploratory_adsorbents`
column for compatibility; the app screens with it set to false.

Comparable database scores retain their existing routing and evidence caveats,
including for products without a training record. A mixture product is retained
only when every entered PFAS has an admissible score; no compound is omitted from aggregation. Related
records cannot override the ML gate or supply a ranking score.

Version-4 bundles optionally carry `adsorbent_training_support.csv` and an
`adsorbent_support` manifest section with counts, policy, and file fingerprint.
The exporter and self-contained packaging hook create the snapshot from actual
fitted records and study bibliography; no model weights are refitted. Packaging
can read DOI metadata from `Standardized_Full`, or use the bundled reference-dose
bibliography when the original workbook is unavailable. Runtime screening uses
only the portable support snapshot. A legacy bundle without this snapshot can
still use comparable database evidence; it cannot verify training identities,
so its ML predictions are not ranked until repackaged. Historical results remain readable.

Result schema 5 adds candidate/per-PFAS `adsorbent_support`, top-level
`adsorbent_screening`, and mixture limiting-score diagnostics. CSV outputs add
product source/study counts and exploratory status. Scenario bundle fingerprints
cover the new snapshot and support implementation.

## Evidence behind each recommendation

The evidence dialog opened by selecting a row distinguishes comparable experimental
evidence from related same-PFAS, same-product records. Comparable records retain
the database-first ranking behavior. Related records fail at least one condition
requirement or have an unreported condition; their values never supply a ranking
score or enter the study-balanced aggregate.

When related records exist, a short count appears inside the candidate's
collapsed details. **View studies and condition differences** opens an optional
popover with up to three representative records, citations, source links, and
the observed/requested conditions that prevented comparison. Counts cover the
full related pool. Examples favor condition agreement and study coverage, never
higher logKd. No general database browser, sidebar filters, or ranking columns
are added. Mixture entries retain evidence separately for every PFAS.

Schema-5 results add `database_evidence.related_evidence` and
`identity_record_counts`; older results remain readable. Run workbooks retain
comparable/related counts, related study counts, a condition summary, and bounded
representative records rather than every related source row.

Logarithmic condition tolerances match a reported zero with a requested zero.
Zero never matches a positive or unreported value; positive factor-of-two
tolerances are unchanged. Explicit pH and TDS zero values survive normalization.
The established unspecified-water assumption remains **ultrapure water**, now
identified in the existing selector and scenario summaries.

## Structured inorganic additives

Inside **Inorganic chemistry**, select an inorganic **Species**, enter its
**Concentration**, choose a **Unit**, then select **Add**. Add multiple species
and remove individual entries. Common salts are suggested; recognized chemical
names/formulas can also be entered in the species selector. Units are mM, µM,
mol/L, mg/L, µg/L, and g/L. Duplicate species (including aliases), unsupported
formulas, and nonpositive/nonfinite concentrations are rejected while preserving
the existing entries. To edit an addition, remove it and add the corrected value.

The shared database normalizer receives tokens assembled from these fields;
there is no free-text additive input or legacy text import. It infers ions and
estimates ionic strength as `0.5 * sum(c * z²)` using its existing dissociation,
charge-balance, and pH-adjuster rules. Known ion selections supplement inferred
ions; an unchecked box cannot erase an ion supplied by a listed salt. Mass
units require a known molar mass. The estimate covers only listed additives,
so unreported background salts are excluded. An entered ionic-strength value
takes precedence, with a visible warning when the additive-derived value
disagrees. The live preview and run workbook expose the chemistry used. Database
comparisons also check derived ionic strength and inferred ions.

Background salts in unspecified or ultrapure water are normalized as **synthetic
water**, following the database convention. Other selected matrices, such as
groundwater, remain selected. Without additives, the ultrapure default remains.
Structured entries and their units persist through reruns and workbook reloads.

## Water-matrix support warnings

Water warnings appear only for the selected, normalized matrix when it has
limited support and the displayed results contain model-derived scores.
Under the current fitted cohort this flags groundwater, tap water, surface
water, wastewater, AFFF solution, and `others`. Ultrapure and synthetic water
do not trigger this warning. Predictions and ranking scores remain available.

The warning appears once above results. **Water-matrix support details** shows
model rows, distinct source records, the largest study ID, the number of studies also
reporting another matrix, and concentration in the largest study. Overlap at
study level does not establish a controlled comparison. Comparable database
scores retain their own evidence-specific caveats; the new warning accompanies
model components in mixtures, saved runs, comparison notes, and workbooks.

`water_support.py` exports `water_training_support.json` from actual fitted
records. It counts source records by dataset and source ID, excludes held-out
prediction rows, and ignores prediction errors. Optional version-4 bundle
metadata fingerprints the snapshot; older bundles without it continue working.
Future bundle exports and the existing repackaging command reuse this hook.
No weights are refitted. Snapshot/policy changes participate in scenario
provenance and stale-result detection.

The documented display heuristics flag fewer than 6 studies, fewer than 50
source records, fewer than 3 studies also containing another matrix, at least
80% of source records from one study, or the broad `others` category. These
are qualitative warning rules, not calibrated confidence or error thresholds.
Passing them does not establish reliable prediction accuracy. Quantitative
claims require separate study-held-out evaluation stratified by water type.

## Organic chemistry

Enter measured **Organic carbon** separately and check **Organic additives
present** when organic material was added. The interface stores a boolean
`organic_matter_present` flag without requesting an organic composition or
additive dose. Carbon concentration is never derived from that flag. Checking
it promotes unspecified/ultrapure water to synthetic water before defaults
are applied; other matrices and supplied carbon measurements are retained.
Missing carbon remains missing when an organic addition is reported.

Marked presence is required in comparable database evidence, consistently
with checked ion indicators. An unchecked control imposes no positive-presence
requirement. The flag persists in saved runs and workbook inputs.

## Download, reload, and compare runs

After screening, **Download run (.xlsx)** creates one durable workbook instead
of separate input and result files. Its sheets contain a summary, the complete
ranked adsorbent table, one row per adsorbent–PFAS pair with evidence and
caveats, and reloadable inputs with original concentration units, structures,
temperature, water chemistry, additives, and ion selections.

Use **Keep for comparison** beside the displayed results to name a completed
run. Kept runs belong to the current browser session and appear in the compact
**Saved runs** control above the inputs only after at least one exists. A kept
run can be reopened, downloaded as the same workbook, or removed. There are at
most 12 runs and 60 MB of saved records per session; duplicate names receive a
suffix and existing runs are never silently overwritten.

Use **Load a previous run** above the input cards to upload an exported `.xlsx`
workbook. The app previews its PFAS and shared chemistry before replacing the
editor. It restores inputs only and rechecks molecular identities against the
currently loaded bundle; historical scores remain reference information inside
the workbook. Screen again to generate current results. Invalid files leave the
editor and saved-run library unchanged.

PFAS-only CSV import remains next to the PFAS editor under **Add PFAS from CSV**.
It appends PFAS identities and concentrations without replacing water settings.
The example file is `examples/pfas_mixture_example.csv`.

After keeping two completed runs, choose **Compare two runs** in **Saved runs**.
Comparison opens as a dedicated view with an input table followed by one aligned
ranking table. Columns are paired by measure—Run A rank, Run B rank, Run A
logKd, Run B logKd, then B − A—before the paired evidence basis and limiting
PFAS columns. Rows join on `Product_key`; products absent from one saved ranking
remain visible with blank scores. The comparison downloads as a separate `.xlsx`
workbook. Full PFAS support and caveats remain inspectable below the table.

Score differences are shown only when the bundle fingerprints and normalized
PFAS structural identities match. Different mixtures or bundles can still be
viewed together with qualitative warnings. Old recommendation exports lack a
bundle fingerprint, so score differences are withheld until inputs are screened
again. Fingerprints include model/data artifacts, prediction code,
descriptor-generator version, and runtime library versions. Bundles using
external data without complete snapshots also withhold score differences.
Comparison does not imply statistical significance,
calibrated uncertainty, or competitive adsorption between mixture components.

## Concentration units and CSV mixtures

Individual PFAS entries support ng/L, µg/L, and mg/L. Changing an entry's unit
rescales its displayed number and preserves the physical concentration. The
service still uses `initial_concentration_mg_l`; scenario exports retain the
entered value and unit under `concentration_input`.

Open **Add PFAS from CSV** in the Target PFAS card and choose **Download example CSV** for
a working four-compound example. The same file is in
`examples/pfas_mixture_example.csv`:

```csv
pfas,identity_type,concentration,unit,parent_smiles
PFOA,abbreviation,100,ng/L,
PFOS,abbreviation,0.25,ug/L,
O=C(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F,smiles,0.0001,mg/L,
375-22-4,cas,75,ng/L,
```

The required columns are `pfas`, `concentration`, and `unit`; column names are
case-insensitive and can appear in any order. The example places `identity_type`
beside the `pfas` value it describes; the preview uses the same grouping.
Optional columns are `identity_type`, `parent_smiles`, and `label`.
`identity_type` accepts `name`, `abbreviation`, `cas`, and `smiles`, and defaults
to `name`, which also matches cached abbreviations. For SMILES input, put the
structure directly in `pfas` and set `identity_type` to `smiles` (the third row
above is PFHxA). Leave `parent_smiles` empty in this case. Accepted units are
ng/L, mg/L, and ug/L, µg/L, or μg/L.

Save as CSV UTF-8, use a decimal point (scientific notation is supported),
and upload at most 100 PFAS rows in a file below 1 MB. Example concentrations
are illustrative. ND, zero, inequalities, nonfinite numbers, and missing or
unknown units are flagged rather than assigned a concentration.

The preview shows every CSV row, its canonical concentration, resolved
identity/structure, and any action needed. All rows must pass before the
complete file is appended to the existing mixture. Duplicate aliases or
equivalent structures are flagged without summing or overwriting entries.
Unknown names/CAS identifiers require parent SMILES in the optional `parent_smiles`
column or resolution through the manual editor; CSV import does not perform
unconfirmed public name lookups. This supplemental structure lets `pfas` retain
the submitted name or CAS number. The previous `smiles` column name is accepted
as an alias for `parent_smiles`; conflicting structures are flagged.
Supplied structures require confirmation before import. Clear the uploaded
file to start a fresh import.

The model ensemble, catalog eligibility, database-first routing, and maximin
mixture ranking are unchanged. A structure fallback supplies only PFAS model
inputs; arbitrary user labels never enter database identity matching.

## Identity and features

- `pfas_descriptors.py` contains the structural counts and sulfur/phosphorus-aware
  TPSA shared with `scripts/PFAS/generate_pfas_features.py`. Batch generation
  retains its full descriptor bank; live generation skips unused longest-path
  descriptors to bound graph-search cost on arbitrary cyclic inputs.
- `pfas_identity.py` returns up to 20 PubChem candidates, with bounded requests,
  a process-wide request limiter, and an hourly bounded cache. Results require
  explicit structure selection and confirmation in the UI. No LLM guesses a
  molecular structure from a name.
- `pfas_cache.py` preserves exact curated values, matches equivalent structures,
  and validates only selected model inputs. New structures use the curated
  Normalizer → Reionizer → Uncharger convention; permanent charges are retained.
  There is no pH-specific speciation calculation. Generated values remain
  separate from the reviewed workbook.

## Service contract

Existing `pfas.identity` / `pfas.identity_type` and `pfas_smiles` callers continue
to work. Uncached SMILES can generate features when the bundle capability is
enabled. To use a confirmed public lookup while retaining its submitted name:

```json
{"pfas": {
  "identity": "submitted name or CAS", "identity_type": "name",
  "confirmed_smiles": "O=C(O)C(F)(F)F", "structure_confirmed": true,
  "label": "Optional display label",
  "resolution": {"source": "pubchem", "cid": 123, "query": "submitted name or CAS"}
}, "initial_concentration_mg_l": 0.1}
```

The service performs no public lookup itself. A confirmed structure is matched
to the curated cache before fallback generation; only this verified cache
match can supply database aliases. In schema 5, `pfas_lookup.ready` is
independent of cache status. `feature_source`, `descriptor_version`,
`rdkit_version`, `identity_resolution`, and `support` retain provenance.
Support includes training membership, the three nearest training structures,
and selected descriptors outside development ranges. Every mixture result
retains the corresponding lookup and support per PFAS, including whether its
limiting PFAS is unseen. Unresolved compounds block the aggregate result and
are reported individually.

## Bundle and validation

Version-4 manifests may optionally carry `pfas_runtime` and the
`unseen_pfas_prediction` capability. `pfas_training_reference.csv` links actual
fitted records to curated structures and aliases. Packaging verifies selected
descriptor parity against the snapshot and records exact generator/RDKit
versions. Missing references, unsupported inputs, or failed parity leave
generation disabled while cached screening remains available. A runtime
version mismatch also disables generation; repackage and revalidate before
using a different RDKit version.

The bundled `pfas_features.xlsx` keeps only the rows of PFAS in the fitted
records, plus the generation audit; the manifest's `pfas_features` section
records `source_rows`, `rows` and the source checksum. An older bundle that
carries the whole shared workbook is trimmed in place, from its own snapshot
so the fitted values are kept, with
`python -m adsorbent_recommender.bundle <bundle-dir> --refresh-pfas-features`
(run from `scripts`), which also repeats the descriptor parity check.

Run tests from the repository root:

```bash
python -m pytest scripts/adsorbent_recommender/tests -q
```

The optional deployment tests skip when its model bundle is absent. The
implementation workspace additionally retains the original service snapshot
for a full-catalog ranking regression. To reproduce the archive audit:

```bash
python scripts/adsorbent_recommender/audit_pfas_holdout.py <split-archive> <bundled-pfas-workbook> <audit-output>
```

The audit uses outer testing predictions, checks name/structure/source-row
separation, and reports per-compound errors and coverage. Its structure mapping
uses the curated deployment snapshot because the archive has no frozen
structure snapshot. Generalization metrics and structural similarity do not
validate an individual prediction, a commercial-adsorbent ranking, or treatment
performance at a site. Ensemble SD remains model disagreement rather than a
calibrated interval.
