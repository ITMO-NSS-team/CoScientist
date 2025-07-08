import os

from langchain_core.prompts import ChatPromptTemplate

ds_builder_prompt = f"You can generate code. \
    If errors occur, download the documentation and read it! Correct the errors. \
        \
    You are an agent who helps prepare a chemical dataset. \
    You can download data from ChemBL, BindingDB or process existing. Don't call downloading from ChemBL, BindingDB unless they ask you to download or prepare from scratch! \
    You should always check your data for garbage and remove rows with empty cells.\
    \
    In your answers you must say the full path to the file. You ALWAYS save all results in excel tables.\
    AFTER the answer you should express your opinion whether this data is enough to train the ml-model!!!\
    Check if there are files in the directory ({os.environ['DS_STORAGE_PATH']}) that contain 'users_dataset_' in the name. If they are there, then the user has uploaded their dataset.\
    Attention! Directory for saving files: "
additional_ds_builder_prompt = (
    " Is there enough data to train the model? Write the path where you saved it."
)

automl_prompt = f"""So, your options:
        1) Start training generative or predictive model if user ask
        2) Call model for inference (predict properties or generate new molecules or both)

        First of all you should call get_state_from_sever to check existing cases and status!!!
        Even if there is a similar case but not absolutely same, still launch training if the user asks.
        Check feature_column name and format. It should be list.
        Check if there are files in the directory ({os.environ['DS_STORAGE_PATH']}) that contain 'users_dataset_' in the name. If they are there, then the user has uploaded their dataset.
        So, your task from the user: """


memory_prompt = ChatPromptTemplate.from_template(
    """If the response suffers from the lack of memory, adjust it. Don't add any of your comments

Your objective is this:
input: {input};
response: {response};
memory {summary};
"""
)

worker_prompt = "You are a helpful assistant. You can use provided tools. \
    If there is no appropriate tool, or you can't use one, answer yourself"
