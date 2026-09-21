# extract.py
import os
import re
import yaml
from jinja2 import Environment, Template, meta
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from chains.enumerate_chain import yaml_cached  # reuse your caching loader
from schemas import ADSORBENT_FIELDS, PERFORMANCE_FIELDS, EXPERIMENT_FIELDS, REVIEW_FIELDS

# Toggle: prune example blocks from prompts (default: ON).
# Set PRUNE_EXAMPLES=0 to re-enable examples without code changes.
PRUNE_EXAMPLES = os.getenv("PRUNE_EXAMPLES", "1") != "0"

def _clean_strings(obj):
    """
    Recursively collapse every string in a dict/list into a single line,
    stripping extra whitespace and newlines.
    """
    if isinstance(obj, dict):
        return {k: _clean_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_strings(v) for v in obj]
    if isinstance(obj, str):
        # collapse all internal whitespace/newlines into single spaces, then strip
        return " ".join(obj.split())
    return obj

def load_examples(prompts_dir: str, task: str) -> str:
    """Load examples for the given task from extract_examples.yaml."""
    try:
        examples_cfg = load_config(prompts_dir, "extract_examples.yaml")
    except FileNotFoundError:
        return ""

    task_examples = examples_cfg.get(task, [])
    if not task_examples:
        return ""

    lines = []
    for ex in task_examples:
        ex_id = ex.get("id", "")
        lines.append(f"Example {ex_id}:")
        if ex.get("requested_Performance_id") or ex.get("requested_Adsorbent_id") or ex.get("requested_Experiment_id"):
            rid = (ex.get("requested_Performance_id") or 
                   ex.get("requested_Adsorbent_id") or 
                   ex.get("requested_Experiment_id"))
            lines.append(f"Requested ID(s): {rid}")
        if ex.get("source_text"):
            lines.append("Source text:")
            lines.append(ex["source_text"].strip())
        if ex.get("expected_record"):
            lines.append("Expected record:")
            lines.append(ex["expected_record"].strip())
            lines.append("")
        if ex.get("notes"):                             
            lines.append("NOTES:")                      
            lines.append(ex["notes"].strip()) 
        lines.append("")  # blank line between examples
    return "\n".join(lines).strip()

def _prune_examples(obj):
    """
    Recursively remove example-like keys from nested dict/list structures.
    Targets common keys: 'example', 'examples', 'examples_block',
    'examples_inline', 'micro_examples', 'samples', 'sample'.
    """
    if isinstance(obj, dict):
        drop = {"example", "examples", "examples_block", "examples_inline",
                "micro_examples", "samples", "sample"}
        return {
            k: _prune_examples(v)
            for k, v in obj.items()
            if k not in drop
        }
    if isinstance(obj, list):
        return [_prune_examples(v) for v in obj]
    return obj


def _render_schema_fields_section(fields: dict) -> str:
    """
    Build the Section III.1 list of field names only (no Python types).
    The LLM does not need ': str' annotations in the prompt.
    """
    # preserve insertion order from schemas.py
    return "\n".join(f"- {k}" for k in fields.keys()).rstrip()


def load_config(prompts_dir: str, filename: str, key: str | None = None):
    """
    Load and cache a YAML file, then return either the full dict or the
    value at a top‐level key.
    """
    cfg = yaml_cached(os.path.join(prompts_dir, filename))
    return cfg[key] if key is not None else cfg

def render_jinja_from_yaml(prompts_dir: str, yaml_file: str, j2_file: str, extra_ctx: dict = {}) -> str:
    """
    Load a YAML context then render a Jinja template with it.
    """
    cfg = load_config(prompts_dir, yaml_file)  
    # read the raw Jinja template
    template_src = open(os.path.join(prompts_dir, j2_file), encoding="utf-8").read()
    tpl = Template(template_src)
    # render *only* the keys your template actually uses, plus any extras like "task"
    return tpl.render(
        extraction_scope=cfg["extraction_scope"],
        apply_rules=    cfg["apply_rules"],
        **extra_ctx
    ).strip()


def _dynamic_template_inputs(template_src: str, partial_variables: dict, candidates: list[str]) -> list[str]:
    """Return dynamic variables used by a Jinja template, excluding partials."""
    parsed = Environment().parse(template_src)
    used = meta.find_undeclared_variables(parsed)
    return [name for name in candidates if name in used and name not in partial_variables]

def render_detail(name: str, detail_dict: dict) -> list[str]:
    """
    Render a single field’s clarification (description + allowed_values +
    shortlist_by_category + example + additional_rules + fallback note)
    into a list of prompt lines.
    """
    lines = [f"- **{name}**: {detail_dict.get('description', detail_dict.get('note', '')).strip()}"]

    # allowed_values
    allowed = detail_dict.get("allowed_values")
    if isinstance(allowed, (list, dict)):
        lines.append("  • allowed_values:")
        if isinstance(allowed, dict):
            for k, v in allowed.items():
                lines.append(f"    - `{k}`: {v}")
        else:
            for v in allowed:
                lines.append(f"    - {v}")

    # shortlist_by_category
    shortlist = detail_dict.get("shortlist_by_category")
    if isinstance(shortlist, dict):
        lines.append("  • shortlist_by_category:")
        for cat, items in shortlist.items():
            lines.append(f"    - {cat}:")
            for it in items:
                lines.append(f"      - {it}")

    # example block (optional; pruned if PRUNE_EXAMPLES)
    if not PRUNE_EXAMPLES:
        example = detail_dict.get("example")
        if example:
            lines.append("  • example:")
            for l in example.strip().splitlines():
                lines.append(f"    {l}")

    # additional_rules
    rules = detail_dict.get("additional_rules")
    if isinstance(rules, list):
        lines.append("  • additional_rules:")
        for r in rules:
            lines.append(f"    - {r}")

    # fallback note
    if "note" in detail_dict and not detail_dict.get("description"):
        lines.append(f"  • note: {detail_dict['note'].strip()}")

    return lines


def _clarification_applies_to_fields(name: str, field_names: set[str]) -> bool:
    """Return whether a clarification key should be rendered for this schema."""
    if name in field_names:
        return True

    if name.casefold().startswith("general note on unit fields"):
        return any(field.endswith("_unit") for field in field_names)

    parts = [
        part.strip()
        for part in re.split(r"\s*(?:,|\band\b)\s*", name)
        if part.strip()
    ]
    return len(parts) > 1 and all(part in field_names for part in parts)


def _render_clarification_section(grouped_cfg: dict[str, dict], fields: dict) -> str:
    """Render only clarifications whose field names belong to the active schema."""
    field_names = set(fields) | set(REVIEW_FIELDS)
    clar_lines: list[str] = []
    section_idx = 1

    for fmt, fields_dict in grouped_cfg.items():
        field_lines: list[str] = []
        for fname in sorted(fields_dict):
            if _clarification_applies_to_fields(fname, field_names):
                field_lines.extend(render_detail(fname, fields_dict[fname]))
        if not field_lines:
            continue
        clar_lines.append(f"{section_idx}. {fmt}")
        clar_lines.extend(field_lines)
        clar_lines.append("")
        section_idx += 1

    return "\n".join(clar_lines).rstrip()


def _build_prompt_sections(prompts_dir: str, task: str, fields: dict) -> tuple[str, str, str, str]:
    """Build prompt sections shared by extraction templates."""
    guidelines = "\n".join(
        render_node(load_config(prompts_dir, "output_format.yaml", "output_format"), indent=0)
    )
    data_scope = render_jinja_from_yaml(prompts_dir, "data_scope.yaml", "data_scope.j2", {"task": task})
    clar_root = load_config(prompts_dir, "clarifications.yaml", "Field_clarifications")
    clar_task = clar_root.get(task, {})
    clar_review = clar_root.get("review", {})
    grouped_cfg: dict[str, dict] = {}
    for fmt, fields_dict in clar_review.items():
        grouped_cfg.setdefault(fmt, {}).update(fields_dict)
    for fmt, fields_dict in clar_task.items():
        grouped_cfg.setdefault(fmt, {}).update(fields_dict)

    clar_sec = _render_clarification_section(grouped_cfg, fields)
    schema_fields_sec = _render_schema_fields_section(fields)
    return guidelines, data_scope, clar_sec, schema_fields_sec

def _performance_value_fields() -> dict:
    """Field map expected by the performance value prompt."""
    return {**PERFORMANCE_FIELDS, **REVIEW_FIELDS}

def render_node(node, indent=0):
    """
    Recursively render a dict or str from YAML into lines of text, preserving keys as headers.
    """
    lines = []
    space = " " * indent

    if isinstance(node, str):
        # block of text
        for l in node.strip().splitlines():
            lines.append(f"{space}{l}")

    elif isinstance(node, dict):
        for key, val in node.items():
            lines.append(f"{space}{key}:")
            # indent the value
            lines.extend(render_node(val, indent + 2))

    else:
        # fallback for lists or other types
        for item in node:
            lines.append(f"{space}- {item}")

    return lines
   

def create_extraction_chain(
    llm,
    prompts_dir: str,
    task: str,
    fields: dict,
    batch_size: int
) -> LLMChain:

    """Build an LLMChain for extracting task records using a file‑based Jinja template."""
    # A) Build IV. Extraction Instructions (from YAML → text)
    guidelines = "\n".join(
        render_node(load_config(prompts_dir, "output_format.yaml", "output_format"), indent=0)
    )

    # B) Build II. Extraction Scope (Jinja render from YAML)
    data_scope = render_jinja_from_yaml(prompts_dir, "data_scope.yaml", "data_scope.j2", {"task": task})

    # C) Load unique IDs & clarifications
    uid_dict  = load_config(prompts_dir, "unique_identifiers.yaml", "unique_identifiers")
    clar_root = load_config(prompts_dir, "clarifications.yaml",    "Field_clarifications")
    try:
        uids_cfg = uid_dict[task]
    except KeyError:
        lookup = {k.lower(): k for k in uid_dict}
        real_key = lookup.get(task.lower())
        if not real_key:
            raise KeyError(f"Task '{task}' not found in unique_identifiers.yaml")
        uids_cfg = uid_dict[real_key]

    clar_task   = clar_root.get(task, {})
    clar_review = clar_root.get("review", {})
    grouped_cfg: dict[str, dict] = {}
    for fmt, fields_dict in clar_review.items():
        grouped_cfg.setdefault(fmt, {}).update(fields_dict)

    for fmt, fields_dict in clar_task.items():
        grouped_cfg.setdefault(fmt, {}).update(fields_dict)
         

    clar_sec = _render_clarification_section(grouped_cfg, fields)
    schema_fields_sec = _render_schema_fields_section(fields)
         

    id_field = uids_cfg["field"]

    # id definition YAML (clean -> optionally prune examples -> dump)
    clean_uids = _clean_strings(uids_cfg)
    if PRUNE_EXAMPLES:
        clean_uids = _prune_examples(clean_uids)
    uid_def = yaml.safe_dump(
        clean_uids,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=1000
    ).rstrip()
    examples_sec = load_examples(prompts_dir, task)

    # D) Load the external Jinja template and wire partials
    template_src = open(os.path.join(prompts_dir, "extract_template.j2"), encoding="utf-8").read()
    prompt = PromptTemplate(
        input_variables=["enriched_text", "figure_captions", "record_to_extract"],
        partial_variables={
            "task": task,
            "id_field": id_field,
            "extraction_scope": data_scope,
            "uid_def": uid_def,
            "clar_sec": clar_sec,
            "guidelines": guidelines,
            "schema_fields_sec": schema_fields_sec,  
            "examples_sec": examples_sec,
        },
        template=template_src,
        template_format="jinja2",
    )
    return LLMChain(llm=llm, prompt=prompt)


def create_study_level_experiment_chain(llm, prompts_dir: str) -> LLMChain:
    """Build the new study-level experiment extraction chain."""
    fields = {**EXPERIMENT_FIELDS, **REVIEW_FIELDS}
    guidelines, data_scope, clar_sec, schema_fields_sec = _build_prompt_sections(
        prompts_dir,
        "experiment",
        fields,
    )
    template_src = open(os.path.join(prompts_dir, "extract_experiment_template.j2"), encoding="utf-8").read()
    partial_variables = {
        "extraction_scope": data_scope,
        "clar_sec": clar_sec,
        "guidelines": guidelines,
        "schema_fields_sec": schema_fields_sec,
    }
    prompt = PromptTemplate(
        input_variables=_dynamic_template_inputs(
            template_src,
            partial_variables,
            [
                "enriched_text",
                "experiment_chunks",
                "figure_captions",
                "water_type_lexicon",
                "water_type_list",
                "adsorbent_lexicon",
                "adsorbent_list",
            ],
        ),
        partial_variables=partial_variables,
        template=template_src,
        template_format="jinja2",
    )
    return LLMChain(llm=llm, prompt=prompt)


def create_study_level_adsorbent_chain(llm, prompts_dir: str) -> LLMChain:
    """Build the study-level adsorbent extraction chain."""
    fields = {**ADSORBENT_FIELDS, **REVIEW_FIELDS}
    guidelines, data_scope, clar_sec, schema_fields_sec = _build_prompt_sections(
        prompts_dir,
        "adsorbent",
        fields,
    )
    template_src = open(os.path.join(prompts_dir, "extract_adsorbent_template.j2"), encoding="utf-8").read()
    partial_variables = {
        "extraction_scope": data_scope,
        "clar_sec": clar_sec,
        "guidelines": guidelines,
        "schema_fields_sec": schema_fields_sec,
    }
    prompt = PromptTemplate(
        input_variables=_dynamic_template_inputs(
            template_src,
            partial_variables,
            [
                "enriched_text",
                "adsorbent_chunks",
                "figure_captions",
                "adsorbent_lexicon",
                "adsorbent_list",
            ],
        ),
        partial_variables=partial_variables,
        template=template_src,
        template_format="jinja2",
    )
    return LLMChain(llm=llm, prompt=prompt)

def create_performance_value_chain(llm, prompts_dir: str) -> LLMChain:
    """Build the chunk-level performance extraction chain for numeric metric records."""
    fields = _performance_value_fields()
    guidelines, _data_scope, clar_sec, schema_fields_sec = _build_prompt_sections(
        prompts_dir,
        "performance",
        fields,
    )
    template_src = open(os.path.join(prompts_dir, "extract_performance_value_template.j2"), encoding="utf-8").read()
    partial_variables = {
        "clar_sec": clar_sec,
        "guidelines": guidelines,
        "schema_fields_sec": schema_fields_sec,
    }
    prompt = PromptTemplate(
        input_variables=_dynamic_template_inputs(
            template_src,
            partial_variables,
            [
                "enriched_text",
                "figure_captions",
                "record_to_extract",
                "water_type_lexicon",
                "water_type_list",
                "adsorbent_lexicon",
                "adsorbent_list",
            ],
        ),
        partial_variables=partial_variables,
        template=template_src,
        template_format="jinja2",
    )
    return LLMChain(llm=llm, prompt=prompt)


def create_performance_unit_chain(llm, prompts_dir: str) -> LLMChain:
    """Build the performance extraction chain for unit-only records."""
    template_src = open(os.path.join(prompts_dir, "extract_performance_unit_template.j2"), encoding="utf-8").read()
    prompt = PromptTemplate(
        input_variables=_dynamic_template_inputs(
            template_src,
            {},
            [
                "enriched_text",
                "figure_captions",
                "requested_unit_fields",
            ],
        ),
        partial_variables={},
        template=template_src,
        template_format="jinja2",
    )
    return LLMChain(llm=llm, prompt=prompt)

def create_extraction_chains(
    prompts_dir: str,
    llm,
    batch_size: int,
    enabled_tasks: set[str] | None = None,
) -> dict:
    enabled = enabled_tasks or {"adsorbent", "performance", "experiment"}
    chains = {}
    if "adsorbent" in enabled:
        chains["adsorbent"] = create_extraction_chain(
            llm, prompts_dir, "adsorbent", ADSORBENT_FIELDS, batch_size
        )
    if "adsorbent_study" in enabled:
        chains["adsorbent_study"] = create_study_level_adsorbent_chain(llm, prompts_dir)
    if "performance" in enabled or "performance_value" in enabled:
        chains["performance_value"] = create_performance_value_chain(llm, prompts_dir)
    if "performance" in enabled or "performance_unit" in enabled:
        chains["performance_unit"] = create_performance_unit_chain(llm, prompts_dir)
    if "performance" in enabled and "performance_value" in chains:
        chains["performance"] = chains["performance_value"]
    if "experiment" in enabled:
        chains["experiment"] = create_extraction_chain(
            llm, prompts_dir, "experiment", EXPERIMENT_FIELDS, batch_size
        )
    if "experiment_study" in enabled:
        chains["experiment_study"] = create_study_level_experiment_chain(llm, prompts_dir)
    return chains
