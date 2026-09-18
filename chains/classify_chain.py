import os
from jinja2 import Template
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from chains.enumerate_chain import yaml_cached
from chains.extract_chain import load_config, render_jinja_from_yaml

def make_classification_chain(
    llm,
    prompts_dir: str,
) -> LLMChain:
    """
    Build an LLMChain for multi‐label classification (performance, adsorbent,
    experiment, or irrelevant), reusing the shared data_scope and
    per‐label definitions.
    """
    # 1) Load & render the universal scope (from data_scope.yaml + data_scope.j2)
    universal_scope = render_jinja_from_yaml(
        prompts_dir,
        "data_scope.yaml",
        "data_scope.j2",
        {"task": "classify"}, 
    )

    # 2) Load the per-label definitions + examples from YAML
    #    Expects prompts_dir/classification/label_defs.yaml
    label_defs = load_config(
        prompts_dir,
        "label_defs.yaml",
        key=None,
    )

    # 3) Read the classification template
    tpl_path = os.path.join(prompts_dir, "classify_template.j2")
    raw_tpl = open(tpl_path, encoding="utf-8").read()

    # 4) Build the PromptTemplate, letting LangChain/Jinja handle everything
    prompt = PromptTemplate(
        input_variables=["title_abstract", "enriched_text"],
        partial_variables={
            "universal_scope": universal_scope,
            "label_defs": label_defs,
        },
        template=raw_tpl,
        template_format="jinja2",
    )

    return LLMChain(llm=llm, prompt=prompt)


def create_classification_chain(prompts_dir: str, llm) -> LLMChain:
    """
    Factory to create the single classification chain.
    """
    return make_classification_chain(llm, prompts_dir)
