import json
import os
import subprocess
import sys
import time
from multiprocessing import Process
from typing import List, Tuple, Union

import pandas as pd
import requests
from smolagents import tool

from ChemCoScientist.tools.utils import filter_valid_strings

# TODO: get from load_env
conf = {"url_pred": "http://10.64.4.247:81", "url_gen": "http://10.32.2.2:94"}


@tool
def get_state_from_server(url: str = "pred") -> Union[dict, str]:
    """Get information about all available models (cases),
    their status (training, trained), description, metrics.

    Important note: if the returned dictionary has the status key not Training, Trained, None, but text content.
    Then an error occurred. And this is its description. Notify the user about it.

    Args:
        url (str): Flag for server, for predictive is 'pred', for generative is 'gen'
    """
    if url == "pred":
        url = conf["url_pred"]
    else:
        url = conf["url_gen"]

    url_ = url.split("http://")[1]
    resp = requests.get("http://" + url_.split("/")[0] + "/check_state")
    if resp.status_code == 500:
        print(f"Server error:{resp.status_code}")
        return "Server error"
    state = json.loads(resp.content)
    return state["state"]


@tool
def get_case_state_from_server(case: str, url: str = "pred") -> Union[dict, str]:
    """Get information about a specific case/model (if found),
    its status (in training, trained), metrics, etc.

    Important note: if the returned dictionary has the status key not Training, Trained, None, but text content.
    Then an error occurred. And this is its description. Notify the user about it.

    Args:
        case (str): Name of case
        url (str): Flag for server, for predictive is 'pred', for generative is 'gen'
    """
    if url == "pred":
        url = conf["url_pred"]
    else:
        url = conf["url_gen"]

    url_ = url.split("http://")[1]
    resp = requests.get("http://" + url_.split("/")[0] + "/check_state")
    if resp.status_code == 500:
        print(f"Server error:{resp.status_code}")
        return "Server error"
    state = json.loads(resp.content)
    try:
        return state["state"][case]
    except:
        return f"Case with name: {case} not found"


@tool
def predict_prop_by_smiles(
    smiles_list: List[str], case: str = "no_name_case", timeout: int = 20
) -> Tuple[requests.Response, dict]:
    """
    Runs property prediction using inference-ready (previously trained) ML models. And RDKIT funcs
    (They need to be calculated using this function, passing feature_column (if the user asks!)is:
    'Validity', 'Brenk', 'QED', 'Synthetic Accessibility', 'LogP', 'Polar Surface Area',
    'H-bond Donors', 'H-bond Acceptors', 'Rotatable Bonds', 'Aromatic Rings',
    'Glaxo', 'SureChEMBL', 'PAINS'.
    They are calculated automatically (by simply func) if they fall into arg: 'feature_column'.

    Args:
        smiles_list (List[str]): A list of molecules in SMILES format.
        case (str, optional): Name of model (model names can be obtained by calling 'get_state_from_server').
        timeout (int, optional): The timeout duration (in minutes).

    Returns:
        Tuple[requests.Response, dict]: A tuple containing the HTTP response object and the parsed JSON response.
    """
    url = conf["url_pred"] + "/predict_ml"
    params = {"case": case, "smiles_list": smiles_list, "timeout": timeout}
    resp = requests.post(url, json.dumps(params))
    return resp, resp.json()


def train_gen_with_data(
    case="no_name",
    data_path="./data_dir_for_coder/kras_g12c_affinity_data.xlsx",  # path to client data folder
    feature_column=["smiles"],
    target_column=[
        "docking_score",
        "QED",
        "Synthetic Accessibility",
        "PAINS",
        "SureChEMBL",
        "Glaxo",
        "Brenk",
        "IC50",
    ],  # All propreties from dataframe you want to calculate in the end
    regression_props=[
        "docking_score"
    ],  # Column name with data for regression tasks (That not include in calculcateble propreties)
    classification_props=[],  # Column name with data for classification tasks (That not include in calculcateble propreties)
    description="Descrption not provided",
    timeout=5,  # min
    url: str = conf["url_gen"] + "/train_gen_models",
    fine_tune: bool = True,
    n_samples=10,
    **kwargs,
):
    """
    Trains a generative deep learning model using user-provided or prepared by a special agent dataset.

    Args:
        case (str): A name of case.
        data_path (str): Path to data for training (in csv or excel format). Must consist SMILES!
        feature_column (list): Names of columns with features (input data) for training. Default is ['smiles'].
        target_column (list): Names of columns (properties) with target data for training. All propreties from dataframe you want to calculate in the end
        regression_props (list): Names of columns with data for regression tasks. Skip if you dont need regression!
        classification_props (list): Names of columns with data for classification tasks. Skip if you dont need classification!
        description (str): Description of model/case.
        timeout (int): Timeout for training in minutes.
        url (str): URL of the server to send the training request to.
        fine_tune (bool): Set alvays to False.
        samples (int): Number of samples for validation. Default is 10.
    """
    start_time = time.time()
    try:
        df = pd.read_csv(
            data_path
        ).to_dict()  # Transfer df to dict for server data transfer
    except:
        df = pd.read_excel(data_path).to_dict()

    params = {
        "case": case,
        "data": df,
        "target_column": target_column,
        "feature_column": feature_column,
        "timeout": timeout,
        "description": description,
        "regression_props": regression_props,
        "classification_props": classification_props,
        "fine_tune": fine_tune,
        "n_samples": n_samples,
        **kwargs,
    }

    p = Process(target=requests.post, args=[url, json.dumps(params)])
    p.start()

    time.sleep(4)
    print("--- %s seconds ---" % (time.time() - start_time))


def train_ml_with_data(
    case="No_name",
    data_path="automl/data/data_4j1r.csv",  # path to client data folder
    feature_column=["Smiles"],
    target_column=[
        "Docking score"
    ],  # All propreties from dataframe you want to calculate in the end,
    regression_props=["Docking score"],
    classification_props=[],
    description="",
    timeout=5,
) -> Union[bool, str]:
    """
    Trains a predictive machine learning model using user-provided or prepared by a special agent dataset.

    This function reads a dataset from a specified file, processes it into a dictionary,
    and sends it to a remote server for training. The training process runs asynchronously
    using a separate process.

    Args:
        case (str, optional): Name of model.
        data_path (str, optional): Path to the CSV file containing the dataset.
        feature_column (list, optional): The name of the column containing the input features. Default is "Smiles".
        target_column (list, optional): All propreties from dataframe you want to calculate in the end.
        regression_props (list, optional): Column names with data for regression tasks. Skip if you dont need regression!
        classification_props (list, optional): Column name with data for classification tasks. Skip if you dont need classification!
        timeout (int, optional): The timeout duration (in minutes) for the request.
        description (str): Description of model/case

    Returns:
        bool (succces or not)"""
    start_time = time.time()
    try:
        df = pd.read_csv(
            data_path
        ).to_dict()  # Transfer df to dict for server data transfer
    except:
        df = pd.read_excel(data_path).to_dict()
    start_time = time.time()
    params = {
        "case": case,
        "data": df,
        "target_column": target_column,
        "feature_column": feature_column,
        "timeout": timeout,
        "description": description,
        "regression_props": regression_props,
        "classification_props": classification_props,
    }

    p = Process(
        target=requests.post,
        args=[f"{conf['url_pred']}/train_ml", json.dumps(params)],
    )
    p.start()

    time.sleep(10)
    p.terminate()

    print("--- %s seconds ---" % (time.time() - start_time))

    return True


def ml_dl_training(
    case: str,
    path: str,
    feature_column=["canonical_smiles"],
    target_column=["docking_score"],
    regression_props=["docking_score"],
    classification_props=[],
):
    ml_ready = False
    train_ml_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
    )
    print("Start training ml model for case: ", case)
    while not ml_ready:
        print("Training ml-model in progress for case: ", case)
        st = get_case_state_from_server(case, "pred")
        if isinstance(st, dict):
            if st["ml_models"]["status"] == "Trained":
                ml_ready = True
        time.sleep(60)

    train_gen_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
        # TODO: rm after testing automl pipeline
        # epoch=1,
    )
    print("Start training gen model for case: ", case)


def ml_dl_training(
    case: str,
    path: str,
    feature_column=["canonical_smiles"],
    target_column=["docking_score"],
    regression_props=["docking_score"],
    classification_props=[],
):
    ml_ready = False
    train_ml_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
    )
    print("Start training ml model for case: ", case)
    while not ml_ready:
        print("Training ml-model in progress for case: ", case)
        st = get_case_state_from_server(case, "pred")
        if isinstance(st, dict):
            if st["ml_models"]["status"] == "Trained":
                ml_ready = True
        time.sleep(60)

    train_gen_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
        # TODO: rm after testing automl pipeline
        # epoch=1,
    )
    print("Start training gen model for case: ", case)


def ml_dl_training(
    case: str,
    path: str,
    feature_column=["canonical_smiles"],
    target_column=["docking_score"],
    regression_props=["docking_score"],
    classification_props=[],
):
    ml_ready = False
    train_ml_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
    )
    print("Start training ml model for case: ", case)
    while not ml_ready:
        print("Training ml-model in progress for case: ", case)
        st = get_case_state_from_server(case, "pred")
        if isinstance(st, dict):
            if st["ml_models"]["status"] == "Trained":
                ml_ready = True
        time.sleep(60)

    train_gen_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
        # TODO: rm after testing automl pipeline
        # epoch=1,
    )
    print("Start training gen model for case: ", case)


@tool
def just_ml_training(
    case: str,
    path: str,
    feature_column: list = ["canonical_smiles"],
    target_column: list = ["docking_score"],
    regression_props: list = ["docking_score"],
    classification_props: list = [],
) -> bool:
    """
    Launch training of ONLY ML-model (predictive).

    Use only as a last resort!

    Args:
        case (str): Name of model.
        path (str): Path to the CSV file containing the dataset.
        feature_column (list): The name of the column containing the input features. You must be sure that such a column exists!
        target_column (list): All propreties from dataframe you want to calculate in the end. This field cannot be left blank (no empty list)!
        regression_props (list, optional): Column names with data for regression tasks. Fill in the list! It should duplicate feature_column.
        classification_props (list, optional): Column name with data for classification tasks. Set '[]' if you dont need classification!
    """

    if regression_props == [] and classification_props == []:
        regression_props = target_column
    if len(target_column) < 1:
        raise ValueError(
            "target_column is empty! You must set value. For example = ['IC50']"
        )
    if len(feature_column) < 1:
        raise ValueError(
            "feature_column is empty! You must set value. For example = ['smiles']"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_excel(path)

    if len(df.values.tolist()) < 300:
        raise ValueError(
            "Training on this data is impossible. The dataset is too small!"
        )

    for column in feature_column:
        if column not in df.columns.tolist():
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )
    for column in target_column:
        if column not in df.columns.tolist():
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )

    # delete molecules eith len more 200
    clear_df = filter_valid_strings(df, feature_column[0])
    if path.split(".")[-1] == "csv":
        clear_df.to_csv(path)
    else:
        clear_df.to_excel(path)

    train_ml_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
    )
    print("Start training ml model for case: ", case)
    return True


@tool
def generate_mol_by_case(
    case: str = "Alzheimer",
    n_samples: int = 10,
) -> dict:
    """Runs molecules generation using inference-ready (previously trained) generative models.

    Args:
        case (str, optional): Name of model (model names can be obtained by calling 'get_state_from_server').
        n_samples (int, optional): Number of molecules to generate. Default is 1
    """
    url = conf["url_gen"] + "/generate_gen_models_by_case"

    params = {
        "case": case,
        "n_samples": n_samples,
    }
    start_time = time.time()
    resp = requests.post(url, data=json.dumps(params))
    print("--- %s seconds ---" % (time.time() - start_time))
    try:
        return json.loads(resp.json())
    except:
        resp.json()


@tool
def run_ml_dl_training_by_daemon(
    case: str,
    path: str,
    feature_column: list = ["smiles"],
    target_column: list[str] = ["docking_score"],
    regression_props: list[str] = ["docking_score"],
    classification_props: list = [],
) -> Union[bool, str]:
    """
    Simultaneously starts training the predictive and generative models (this is normal
    if the user asks for only one thing).
    The status can be checked with "get_state_case_from_server".

    Args:
        case (str): Name of model.
        path (str): Path to the CSV file containing the dataset.
        feature_column (list): The name of the column containing the input features. You must be sure that such a column exists!
        target_column (list): All propreties from dataframe you want to calculate in the end. This field cannot be left blank (no empty list)!
        regression_props (list, optional): Column names with data for regression tasks. Fill in the list! It should duplicate feature_column.
        classification_props (list, optional): Column name with data for classification tasks. Set '[]' if you dont need classification!

    note: Either regression_props or classification_props must be filled in.
    """
    if isinstance(feature_column, str):
        feature_column = [feature_column]

    if regression_props == [] and classification_props == []:
        regression_props = target_column
    if len(target_column) < 1:
        raise ValueError(
            "target_column is empty! You must set value. For example = ['IC50']"
        )
    if len(feature_column) < 1:
        raise ValueError(
            "feature_column is empty! You must set value. For example = ['smiles']"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_excel(path)

    if len(df.values.tolist()) < 300:
        raise ValueError(
            "Training on this data is impossible. The dataset is too small!"
        )

    for column in feature_column:
        if column not in df.columns.tolist():
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )
    for column in target_column:
        if column not in df.columns.tolist():
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )

    # delete molecules eith len more 200
    clear_df = filter_valid_strings(df, feature_column[0])
    if path.split(".")[-1] == "csv":
        clear_df.to_csv(path)
    else:
        clear_df.to_excel(path)

    cmd = [
        sys.executable,
        "-c",
        (
            "from ChemCoScientist.tools.ml_tools import ml_dl_training;"
            "ml_dl_training("
            f"case='{case}',"
            f"path='{path}',"
            f"feature_column={feature_column},"
            f"target_column={target_column},"
            f"regression_props={regression_props},"
            f"classification_props={classification_props}"
            ")"
        ),
    ]

    try:
        subprocess.Popen(
            cmd,
            stdout=open("/tmp/ml_training.log", "a"),
            stderr=open("/tmp/ml_training.err", "a"),
            cwd="/Users/alina/Desktop/ИТМО/ChemCoScientist",
        )
        time.sleep(5)
        return True
    except Exception as e:
        print(f"Failed to start process: {e}", file=sys.stderr)
        return False


agents_tools = [
    run_ml_dl_training_by_daemon,
    get_case_state_from_server,
    get_state_from_server,
    generate_mol_by_case,
    predict_prop_by_smiles,
]
if __name__ == "__main__":

    # get_state_from_server("http://10.32.2.2:81")
    just_ml_training(
        "data_cyk_short_v2",
        regression_props=["IC50"],
        feature_column=["canonical_smiles"],
        target_column=["IC50"],
        path="./data_dir_for_coder/data_cyk_short.csv",
    )
