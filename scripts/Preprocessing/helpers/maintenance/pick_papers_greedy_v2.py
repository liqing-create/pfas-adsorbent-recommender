"""Greedy-coverage selection of the studies used to develop the LLM prompts.

This is the March 2025 script behind the development / validation / application
divide of the studies (moved here from ``Code/old`` on 2026-10-13). It picks
papers that together cover the PFAS and adsorbents in the corpus, then adds
random picks. The final assignment of every study is recorded in
``workflow_config.STUDY_GROUPS`` (copied from ``web of science search/studies.txt``);
the selection saved by this script in 2025 only partly matches it.
"""
import json
import os
import random
import csv
from pathlib import Path

# Project paths are derived from this file:
# <Box>/My Box Notes/2. Adsorbent Recommender/AD/scripts/Preprocessing
RECOMMENDER_ROOT = Path(__file__).resolve().parents[3]
LEGACY_DIR = RECOMMENDER_ROOT / "Code" / "old"

# Input/Output paths
METADATA_FILE = LEGACY_DIR / "metadata_output.json"
# The 2025 selection is kept as Code/old/selected_papers.json; a rerun draws new
# random picks, so it writes next to that file instead of over it.
OUTPUT_SELECTED = LEGACY_DIR / "selected_papers_rerun.json"
CSV_FILE = RECOMMENDER_ROOT / "History data" / "Adsorbent_Processed" / "adsorption_analysis_v1.csv"

# Number of additional random picks you want
NUM_RANDOM_PICKS = 30

def load_csv_parameters(csv_file):
    """
    Load the CSV file and build a dictionary with the study id as key.
    The CSV has columns: study, Langmuir/Freundlich Sentences, Kd Sentences, Parameter.
    """
    paper_params = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            study_id = row["study"].strip()
            parameter = row["Parameter"].strip()
            lf_sentences = row["Langmuir/Freundlich Sentences"].strip()
            kd_sentences = row["Kd Sentences"].strip()
            paper_params[study_id] = {
                "parameter": parameter,
                "LF_sentences": lf_sentences,
                "Kd_sentences": kd_sentences
            }
    return paper_params

def main():
    """
    1) Load metadata_output.json and the CSV file.
    2) Exclude papers with Parameter 'Kd' or 'Removal Rate'.
    3) Perform greedy coverage selection based on the PFAS+adsorbent lists.
    4) Randomly pick additional papers from the remainder.
    5) Ensure the final selection covers both 'Langmuir/Freundlich' and 'Both'.
    6) Merge parameter information into the final output.
    7) Save final selection to selected_papers.json.
    """
    # Load metadata from JSON
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read {METADATA_FILE}: {e}")
        return

    # metadata_output.json was later rewritten by dictionary_search_v2.py with
    # different key names; accept both spellings.
    for item in all_metadata:
        for old_key, new_key in (("study_id", "study"), ("pfas_list", "PFAS_list"), ("adsorbent_list", "Adsorbent_list")):
            if old_key not in item and new_key in item:
                item[old_key] = item[new_key]

    # Load CSV parameters
    try:
        paper_params = load_csv_parameters(CSV_FILE)
    except Exception as e:
        print(f"❌ Failed to read CSV file {CSV_FILE}: {e}")
        return

    # Filter out studies with Parameter 'Kd' or 'Removal Rate'
    filtered_metadata = []
    for item in all_metadata:
        study_id = item.get("study_id")
        param_info = paper_params.get(study_id, {})
        parameter = param_info.get("parameter", "").strip().lower()
        if parameter in ["kd", "removal rate"]:
            continue
        filtered_metadata.append(item)
    
    if not filtered_metadata:
        print("❌ No studies remain after filtering based on parameter.")
        return
    
    all_metadata = filtered_metadata

    # 1) Convert each paper's PFAS+ads list -> single set for greedy coverage
    paper_terms = {}
    all_terms = set()

    for item in all_metadata:
        study_id = item.get("study_id")
        pfas = item.get("pfas_list", [])
        ads = item.get("adsorbent_list", [])
        combined_set = set(pfas) | set(ads)
        paper_terms[study_id] = combined_set
        all_terms |= combined_set

    uncovered_terms = set(all_terms)
    chosen_papers = []

    # 2) Greedy coverage selection
    while uncovered_terms:
        best_paper = None
        best_newly_covered = 0
        for study_id, terms_set in paper_terms.items():
            if terms_set:
                newly_covered = len(terms_set & uncovered_terms)
                if newly_covered > best_newly_covered:
                    best_newly_covered = newly_covered
                    best_paper = study_id
        if best_paper is None:
            print("⚠️ No paper can add new coverage (or we've covered everything).")
            break

        chosen_papers.append(best_paper)
        # Remove covered terms
        uncovered_terms -= paper_terms[best_paper]
        # Mark this paper as used
        paper_terms[best_paper] = set()

        if not uncovered_terms:
            break

    # 3) Random picks from the remainder
    all_paper_ids = [item["study_id"] for item in all_metadata]
    remainder = [pid for pid in all_paper_ids if pid not in chosen_papers]
    random.shuffle(remainder)
    extra_random = remainder[:NUM_RANDOM_PICKS]
    final_papers = chosen_papers + extra_random

    # 4) Ensure coverage of both 'Langmuir/Freundlich' and 'Both'
    # Check the parameter types in the final selection using the CSV info.
    final_params = {}
    for study_id in final_papers:
        if study_id in paper_params:
            param = paper_params[study_id]["parameter"].strip()
            final_params[study_id] = param

    required_categories = ["Langmuir/Freundlich", "Both"]
    missing_categories = []
    for category in required_categories:
        if not any(param == category for param in final_params.values()):
            missing_categories.append(category)
    
    # For each missing category, try to add a candidate from the remainder.
    for category in missing_categories:
        candidate = None
        for study_id in remainder:
            if study_id in paper_params and paper_params[study_id]["parameter"].strip() == category:
                candidate = study_id
                break
        if candidate:
            final_papers.append(candidate)
            print(f"✅ Added paper {candidate} to cover missing category: {category}")
        else:
            print(f"⚠️ No available paper found for missing category: {category}")

    # 5) Summarize final selection
    print(f"\nGreedy coverage picked {len(chosen_papers)} papers.")
    print(f"Randomly picked {len(extra_random)} additional papers.")
    print(f"Total selected before ensuring category coverage: {len(chosen_papers) + len(extra_random)}")
    print(f"Total final selected: {len(final_papers)}")

    # 6) Reconstruct final data from metadata lookup and add parameter info from CSV
    meta_dict = {d["study_id"]: d for d in all_metadata}
    final_data = []
    for study_id in final_papers:
        if study_id in meta_dict:
            study_data = meta_dict[study_id].copy()  # create a copy of the metadata dictionary
            if study_id in paper_params:
                study_data["parameter"] = paper_params[study_id]["parameter"].strip()
            final_data.append(study_data)
        else:
            print(f"⚠️ Study {study_id} not found in metadata.")

    # 7) Save the final selection
    with open(OUTPUT_SELECTED, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2)

    print(f"✅ Final selection saved to {OUTPUT_SELECTED}")

if __name__ == "__main__":
    main()
