import os

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from chains.llm_config import get_llm


PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "prompts",
)


def create_structured_caption_resolution_chain(
    provider: str,
    model_name: str,
) -> LLMChain:
    template_path = os.path.join(
        PROMPTS_DIR,
        "structured_caption_resolution_template.j2",
    )
    with open(template_path, encoding="utf-8") as handle:
        template_src = handle.read()

    prompt = PromptTemplate(
        input_variables=["study_id", "source_file", "snippets_text"],
        template=template_src,
        template_format="jinja2",
    )
    return LLMChain(
        llm=get_llm(provider=provider, model_name=model_name),
        prompt=prompt,
    )
