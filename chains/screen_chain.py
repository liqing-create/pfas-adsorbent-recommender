import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def make_screen_chain(llm, prompts_dir: str) -> LLMChain:
    """
    Abstract-level screening chain (Y/N). The entire prompt lives in prompts/screen.j2.
    Only the ABSTRACT is provided to the model.
    """
    tpl_path = os.path.join(prompts_dir, "screen.j2")
    with open(tpl_path, encoding="utf-8") as fh:
        raw_tpl = fh.read()

    prompt = PromptTemplate(
        input_variables=["abstract"],
        template=raw_tpl,
        template_format="jinja2",
    )
    return LLMChain(llm=llm, prompt=prompt)

def create_screen_chain(prompts_dir: str, llm) -> LLMChain:
    return make_screen_chain(llm, prompts_dir)
