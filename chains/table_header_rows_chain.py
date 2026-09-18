import os

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from chains.llm_config import get_llm


PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "prompts",
)


def create_table_header_rows_chain(
    provider: str,
    model_name: str,
) -> LLMChain:
    template_path = os.path.join(
        PROMPTS_DIR,
        "table_header_rows_template.j2",
    )
    with open(template_path, encoding="utf-8") as f:
        template_src = f.read()

    prompt = PromptTemplate(
        input_variables=[
            "study_id",
            "source_file",
            "table_id",
            "caption_text",
            "docling_header_rows",
            "numbered_rows",
        ],
        template=template_src,
        template_format="jinja2",
    )

    llm = get_llm(
        provider=provider,
        model_name=model_name,
    )

    return LLMChain(llm=llm, prompt=prompt)
