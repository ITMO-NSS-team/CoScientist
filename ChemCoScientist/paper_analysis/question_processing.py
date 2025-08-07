import base64
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from protollm.connectors import create_llm_connector

from ChemCoScientist.paper_analysis.chroma_db_operations import ChromaDBPaperStore
from ChemCoScientist.paper_analysis.prompts import sys_prompt, explore_my_papers_prompt
from CoScientist.paper_parser.utils import convert_to_base64, prompt_func
from definitions import CONFIG_PATH

from ChemCoScientist.frontend.utils import update_activity

load_dotenv(CONFIG_PATH)

VISION_LLM_URL = os.environ["VISION_LLM_URL"]

PAPER_STORE = ChromaDBPaperStore()


def query_llm(
    model_url: str, question: str, txt_context: str, img_paths: list[str]
) -> tuple:
    llm = create_llm_connector(model_url)

    img_context = list(map(convert_to_base64, img_paths))

    messages = [
        SystemMessage(content=sys_prompt),
        prompt_func(
            {
                "text": f"USER QUESTION: {question}\n\nCONTEXT: {txt_context}",
                "image": img_context,
            }
        ),
    ]

    res = llm.invoke(messages)
    return res.content, res.response_metadata


def simple_query_llm(model_url: str, question: str, pdfs: list,) -> dict:
    if pdfs:
        update_activity(os.path.dirname(pdfs[0]))

    llm = create_llm_connector(model_url)

    content = []

    for paper_pdf in pdfs:
        with open(paper_pdf, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        paper_part = {
            "type": "file",
            "file": {
                "filename": paper_pdf,
                "file_data": f"data:application/pdf;base64,{base64_pdf}",
            },
        }
        content.append(paper_part)

    text_part = {"type": "text", "text": f"USER QUESTION: {question}"}
    content.append(text_part)
    from langchain_core.messages import HumanMessage

    messages = [
        SystemMessage(content=explore_my_papers_prompt),
        HumanMessage(content=content)
    ]

    res = llm.invoke(messages)
    return {'answer': res.content}


def process_question(question: str) -> dict:
    txt_data, img_data = PAPER_STORE.retrieve_context(question)
    txt_context = ""
    img_paths = set()

    for idx, chunk in enumerate(txt_data, start=1):
        txt_context += (
            f"{idx}. Metadata: "
            + str(chunk[2])
            + "\nChunk: "
            + chunk[1].replace("passage: ", "")
            + "\n\n"
        )
    for chunk_meta in [chunk[2] for chunk in txt_data]:
        img_paths.update(eval(chunk_meta["imgs_in_chunk"]))
    for img in img_data["metadatas"][0]:
        img_paths.add(img["image_path"])

    ans, metadata = query_llm(VISION_LLM_URL, question, txt_context, list(img_paths))

    return {
        "answer": ans,
        "metadata": {
            "text_context": txt_context,
            "image_context": img_paths,
            "metadata": metadata,
        },
    }


if __name__ == "__main__":
    # file_paths = []  # Enter list of paths to images here
    #
    # images = list(map(convert_to_base64, file_paths))
    #
    # llm = create_llm_connector(VISION_LLM_URL)
    #
    # # question = ("Какая реакция идет протекает на 6 стадии Total Synthesis of (−)-Glionitrin A/B? Какие реагенты"
    # #             " участвовали в реакции и какой продукт получили? Какой получился выход?")
    # question = ("I need all the compounds that were used in the experiments. Obligatorily I need all results to be in"
    #             " the form of a table of 2 columns where in the first column were the names by IUPAC numberclature and"
    #             " in the second column in SMILES notation. Don't add it to this list of reaction products for me. Can"
    #             " you do that?")
    # context = ""
    #
    # messages = [
    #     SystemMessage(content=sys_prompt),
    #     prompt_func({"text": f"USER QUESTION: {question}\n\nCONTEXT: {context}", "image": images})
    # ]
    # # messages = [
    # #     SystemMessage(content="You're a useful assistant. You only ever reply in the form of valid JSON."),
    # #     prompt_func(
    # #         {
    # #             "text": "For the provided images, generate a detailed clear description. If there is a table in the"
    # #                     " image, parse it and return it in HTML format. If you see chemical compounds in the figures,"
    # #                     " output the names of the compounds according to IUPAC nomenclature.\n"
    # #                     " As a response, return ONLY JSON of the following form: {‘figure_1’:"
    # #                     " ‘figure_1_description’, ‘figure_2’: ‘figure_2_description’, ‘table_1’:"
    # #                     " ‘table_1_description’...}",
    # #             "image": images
    # #         }
    # #     )
    # # ]
    #
    # res = llm.invoke(messages)
    # print(res.content)
    # print(res.response_metadata)

    question = 'how does the synthesis of Glionitrin A/B happen?'
    question = 'what is the name of Figure 2 in the given paper?'
    question = 'what is the time in the first line of table 1?'
    question = 'what is the time in the second line of table 1?'
    question = 'what is the elecrtophile in the 8th line of table 1?'

    question = 'what is (R)-13b on scheme 1?'
    question = 'what is depicted on graphs that are on scheme 1? give a detailed description. what conclusions can be made from them?'
    question = 'what compounds are depicted above table 1?'
    question = 'list all 13 chemical compounds that are depicted on scheme 1'
    # res = process_question(question)
    # print(res)

    paper = "/Users/lizzy/Documents/WORK/projects/CoScientist/PaperAnalysis/papers/koning-et-al-2021-total-synthesis.pdf"

    # res = simple_query_llm(VISION_LLM_URL, question, [paper])
    res = process_question(question)
    from pprint import pprint
    pprint(res)
