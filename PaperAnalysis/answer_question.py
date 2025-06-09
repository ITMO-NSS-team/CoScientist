import base64
from io import BytesIO

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from protollm.connectors import create_llm_connector

load_dotenv("../config.env")

sys_prompt = ("You are a helpful chemist assistant. Answer USER QUESTION in a direct tone. Be"
              " moderately concise. Your audience is an expert, so be highly specific. If there are"
              " ambiguous terms or acronyms, first define them. For answer you must use CONTEXT"
              " provided by user. CONTEXT includes text and pictures. Analyze CONTEXT and answer the question."
              "\nRules:\n1. You must use only provided information for the answer.\n2. Add a unit of"
              " measurement to an answer only if appropriate.\n3. For answer you should take only that"
              " information from context, which is relevant to user's question.\n4. If you do not know"
              " how to answer the questions, say so.\n5. If you are additionally given images, you can"
              " use the information from them as CONTEXT to answer.\n 6. Use valid IUPAC or SMILES "
              " notation if necessary to answer the question.")

summarisation_prompt = ("Create a concise and informative summary of the following scientific article. Focus on the key"
                        " elements:\n"
                        "1. Objective : Describe the main problem, hypothesis, or research question addressed.\n"
                        "2. Methodology : Highlight the key methods, experiments, or approaches used in the study.\n"
                        "3. Results : Summarize the primary findings, data, or observations, including statistical"
                        " significance (if applicable).\n"
                        # "4. Significance : Explain how the results contribute to the field, emphasizing practical"
                        # " or theoretical value.\n"
                        #"5. Limitations : Mention weaknesses of the study or conditions that may affect result"
                        #" interpretation.\n"
                        #"6. Future Directions : Note the authors’ recommendations for further research.\n\n"
                        "Maintain a neutral tone, ensure logical flow. Emphasize the novelty of the work and how it"
                        " differs from prior studies. Maximum length: 200 words. Don't add any comments at the"
                        " beginning and end of the summary. Before the summary, indicate on a separate line all"
                        " keywords/terms that characterise the article.\n\n"
                        "Article in Markdown markup:\n")


def convert_to_base64(file_path):
    """
    Convert PIL images to Base64 encoded strings

    :param file_path: path to image
    :return: Re-sized Base64 string
    """
    pil_image = Image.open(file_path)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def prompt_func(data):
    text = data["text"]
    imgs = data["image"]
    content_parts = []
    
    for img in imgs:
        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{img}",
        }
        content_parts.append(image_part)
    
    text_part = {"type": "text", "text": text}
    content_parts.append(text_part)
    
    return HumanMessage(content=content_parts)


def query_llm(model_url: str, question: str, txt_context: str, img_paths: list[str]) -> tuple:
    llm = create_llm_connector(model_url)

    img_context = list(map(convert_to_base64, img_paths))

    messages = [
        SystemMessage(content=sys_prompt),
        prompt_func({"text": f"USER QUESTION: {question}\n\nCONTEXT: {txt_context}", "image": img_context})
    ]

    res = llm.invoke(messages)
    return res.content, res.response_metadata


if __name__ == "__main__":
    file_paths = []  # Enter list of paths to images here

    images = list(map(convert_to_base64, file_paths))

    llm = create_llm_connector("https://api.vsegpt.ru/v1;vis-google/gemini-2.0-flash-001")

    # question = ("Какая реакция идет протекает на 6 стадии Total Synthesis of (−)-Glionitrin A/B? Какие реагенты"
    #             " участвовали в реакции и какой продукт получили? Какой получился выход?")
    question = ("I need all the compounds that were used in the experiments. Obligatorily I need all results to be in"
                " the form of a table of 2 columns where in the first column were the names by IUPAC numberclature and"
                " in the second column in SMILES notation. Don't add it to this list of reaction products for me. Can"
                " you do that?")
    context = ""
    
    messages = [
        SystemMessage(content=sys_prompt),
        prompt_func({"text": f"USER QUESTION: {question}\n\nCONTEXT: {context}", "image": images})
    ]
    # messages = [
    #     SystemMessage(content="You're a useful assistant. You only ever reply in the form of valid JSON."),
    #     prompt_func(
    #         {
    #             "text": "For the provided images, generate a detailed clear description. If there is a table in the"
    #                     " image, parse it and return it in HTML format. If you see chemical compounds in the figures,"
    #                     " output the names of the compounds according to IUPAC nomenclature.\n"
    #                     " As a response, return ONLY JSON of the following form: {‘figure_1’:"
    #                     " ‘figure_1_description’, ‘figure_2’: ‘figure_2_description’, ‘table_1’:"
    #                     " ‘table_1_description’...}",
    #             "image": images
    #         }
    #     )
    # ]
    
    res = llm.invoke(messages)
    print(res.content)
    print(res.response_metadata)
