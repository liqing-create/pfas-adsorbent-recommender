import os
import yaml
from jinja2 import Template
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ── helpers ──────────────────────────────────────────────────────────
def read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()

def load_yaml(path: str):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def render_jinja_from_yaml(prompts_dir: str, yaml_file: str, j2_file: str, extra_ctx: dict | None = None) -> str:
    """
    Load a YAML context then render a Jinja template with it.
    """
    ctx = yaml_cached(os.path.join(prompts_dir, yaml_file))
    if extra_ctx:
        ctx = {**ctx, **extra_ctx}
    tpl = read(os.path.join(prompts_dir, j2_file))
    return Template(tpl).render(**ctx).strip()

def first_existing_template(prompts_dir: str, names: list[str]) -> str:
    for name in names:
        path = os.path.join(prompts_dir, name)
        if os.path.exists(path):
            return path
    expected = ", ".join(os.path.join(prompts_dir, name) for name in names)
    raise FileNotFoundError(f"None of the expected prompt templates exist: {expected}")

_yaml_cache = {}
def yaml_cached(path: str):
    if path not in _yaml_cache:
        _yaml_cache[path] = load_yaml(path)
    return _yaml_cache[path]

def _clean_strings(obj):
    if isinstance(obj, dict):
        return {k: _clean_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_strings(v) for v in obj]
    if isinstance(obj, str):
        return " ".join(obj.split())
    return obj

def _common_parts(prompts_dir: str, task: str):
    # universal scope
    scope_txt = render_jinja_from_yaml(prompts_dir, "data_scope.yaml", "data_scope.j2")

    # unique identifiers
    uid_yml   = yaml_cached(os.path.join(prompts_dir, "unique_identifiers.yaml"))
    uid_dict = uid_yml["unique_identifiers"]

    # case-insensitive lookup
    try:
        uid_block = uid_dict[task]
    except KeyError:
        lookup = {k.lower(): k for k in uid_dict}
        real_key = lookup.get(task.lower())
        if not real_key:
            raise KeyError(f"Task '{task}' not found in unique_identifiers.yaml")
        uid_block = uid_dict[real_key]

    id_field = uid_block["field"]
    clean_block = _clean_strings(uid_block)
    uid_def = yaml.safe_dump(
        clean_block,
        sort_keys=False, allow_unicode=True,
        default_flow_style=False, width=1000
    )
    return scope_txt, uid_def, id_field

# ── planning ─────────────────────────────────────────────────────────

def render_node(node, indent=0):
    lines = []
    space = " " * indent
    if isinstance(node, str):
        for line in node.strip().splitlines():
            lines.append(f"{space}{line}")
    elif isinstance(node, dict):
        for key, val in node.items():
            lines.append(f"{space}{key}:")
            lines.extend(render_node(val, indent + 2))
    elif isinstance(node, list):
        for item in node:
            lines.extend(render_node(item, indent))
    return lines

def render_output_guidelines(prompts_dir: str) -> str:
    cfg = yaml_cached(os.path.join(prompts_dir, "output_format.yaml"))
    node = cfg.get("output_format", cfg)
    return "\n".join(render_node(node, indent=0)).strip()

def create_performance_enumeration_chain(prompts_dir: str, llm) -> LLMChain:
    raw_tpl = read(os.path.join(prompts_dir, "enumerate_performance_template.j2"))
    prompt = PromptTemplate(
        input_variables=[
            "enriched_text",
            "figure_captions",
            "adsorbent_list",
            "water_type_list",
        ],
        partial_variables={
            "guidelines": render_output_guidelines(prompts_dir),
        },
        template=raw_tpl,
        template_format="jinja2",
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

def create_water_type_lexicon_chain(prompts_dir: str, llm) -> LLMChain:
    raw_tpl = read(
        first_existing_template(
            prompts_dir,
            ["enumerate_water_type_template.j2", "enumerate_water_typen_template.j2"],
        )
    )
    prompt = PromptTemplate(
        input_variables=["abstract", "paper_chunks"],
        template=raw_tpl,
        template_format="jinja2",
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

def create_enumeration_chains(prompts_dir: str, llm, task: str, file_type: str):
    task_norm = str(task or "").strip().lower()
    if task_norm == "performance":
        return None, create_performance_enumeration_chain(prompts_dir, llm)
    if task_norm == "adsorbent":
        return None, create_adsorbent_lexicon_chain(prompts_dir, llm)
    if task_norm in {"water_type", "water"}:
        return None, create_water_type_lexicon_chain(prompts_dir, llm)
    raise ValueError(f"Unsupported enumeration task: {task!r}")

def _chunk_text(ch: dict) -> str:
    return (ch.get("enriched_text") or ch.get("text") or "").strip()


def build_study_abstract(chunks: list[dict]) -> str:
    """Return abstract text from main-paper abstract chunks."""
    abstracts = []
    seen = set()
    for ch in chunks or []:
        if not isinstance(ch, dict):
            continue
        if str(ch.get("file_type") or "").strip().lower() != "main_paper":
            continue
        if str(ch.get("section_label") or "").strip().lower() != "abstract":
            continue
        text = _chunk_text(ch)
        if not text:
            continue
        key = " ".join(text.split())
        if key in seen:
            continue
        seen.add(key)
        abstracts.append(text)
    return "\n\n".join(abstracts).strip()


def build_study_chunks(chunks: list[dict]) -> str:
    """Format selected study chunks for study-level prompt injection."""
    lines = []
    n = 0
    for ch in chunks or []:
        text = _chunk_text(ch)
        if not text:
            continue
        n += 1
        file_type = (ch.get("file_type") or "unknown").strip()
        chunk_id = ch.get("chunk_id")
        source_id = f"{file_type}_{chunk_id}" if chunk_id is not None else file_type
        labels = ch.get("predicted_label") or ch.get("annotation") or ""
        if isinstance(labels, list):
            labels = ", ".join(str(x) for x in labels)
        lines.append(f"Chunk {n} (Data_Source={source_id}; labels={labels}):\n{text}")
    return "\n\n".join(lines).strip()


def create_adsorbent_lexicon_chain(prompts_dir: str, llm) -> LLMChain:
    raw_tpl = read(os.path.join(prompts_dir, "enumerate_adsorbent_template.j2"))
    prompt = PromptTemplate(
        input_variables=["abstract", "paper_chunks"],
        template=raw_tpl,
        template_format="jinja2",
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

create_adsorbent_enumeration_chain = create_adsorbent_lexicon_chain
create_water_type_enumeration_chain = create_water_type_lexicon_chain
