import json
import os
import sys
import time
from typing import List, Tuple, Union, Literal, Dict, Any, Optional
from enum import Enum
import pandas as pd
import requests
from smolagents import tool
#from ChemCoScientist.tools.utils import filter_valid_strings
import os
import json
import aiohttp
import asyncio
from collections import defaultdict
import operator
from functools import partial
from pathlib import Path

# from dotenv import load_dotenv

# load_dotenv('/app/config.env')

conf = {"url_pred": f'{os.environ.get("ML_TOOLS_IP")}:{os.environ.get("ML_TOOLS_PORT")}', "url_gen": f'{os.environ.get("DL_TOOLS_IP")}:{os.environ.get("DL_TOOLS_PORT")}'}

async def train_gen_with_data_async(
    case: str ="no_name",
    data_path: str ="./data_dir_for_coder/kras_g12c_affinity_data.xlsx",  # path to client data folder
    feature_column: List[str] = ["smiles"],
    target_column: List[str] = [
        "docking_score",
        "QED",
        "Synthetic Accessibility",
        "PAINS",
        "SureChEMBL",
        "Glaxo",
        "Brenk",
        "IC50",
    ],  # All propreties from dataframe you want to calculate in the end
    regression_props: List[str] =["docking_score"],  # Column name with data for regression tasks (That not include in calculcateble propreties)
    classification_props: List[str] = [],  # Column name with data for classification tasks (That not include in calculcateble propreties)
    description: str = None,
    timeout: int = 5,  # min
    url: str = conf["url_gen"] + "/train_gan",
    fine_tune: bool = True,
    n_samples:int = 10,
    **kwargs,
):
    """
    Trains a generative deep learning model to learn relationships between molecular structures and their properties from a provided dataset.
    This method prepares data from a CSV or Excel file, containing SMILES strings and associated properties, and sends it to a server for model training.
    The trained model can then be used to generate new molecules with desired characteristics.
    
    Args:
        case (str): A name for the training case, used for identification.
        data_path (str): Path to the data file (CSV or Excel) containing molecular data. The file must include SMILES strings.
        feature_column (list): List of column names representing the molecular features (inputs). Defaults to ['smiles'].
        target_column (list): List of column names representing the properties to be predicted (targets).
        regression_props (list): List of column names representing properties to be predicted using regression.
        classification_props (list): List of column names representing properties to be predicted using classification.
        description (str): A description of the model or training case.
        timeout (int): Training timeout in minutes.
        url (str): URL of the server endpoint for training the model. Defaults to a configured URL.
        fine_tune (bool):  A flag indicating whether to fine-tune an existing model. Defaults to True.
        n_samples (int): Number of samples to use for validation during training. Defaults to 10.
        **kwargs: Additional keyword arguments to be passed to the training process.
    
    Returns:
        Dict[str, Any]: Response from the training server containing training results.
    """
    start_time = time.time()

    data_path_obj = Path(data_path)
    ext = data_path_obj.suffix.lower()

    if ext in ('.csv', '.txt', '.tsv'):
        df = pd.read_csv(data_path_obj).to_dict()
    elif ext in ('.xlsx', '.xls', '.xlsm', '.xlsb'):
        df = pd.read_excel(data_path_obj).to_dict()
    elif ext in ('.parquet', '.parq'):
        df = pd.read_parquet(data_path_obj).to_dict()


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

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                json=params,  # aiohttp automatically serializes to JSON
                timeout=aiohttp.ClientTimeout(total=10800)  # 3 hours timeout for the request
            ) as response:
                
                response.raise_for_status()
                result = await response.json()

                print(f"Training completed successfully for case: {case}")
                print(f"--- {time.time() - start_time:.2f} seconds ---")

                return result
     
    except aiohttp.ClientError as e:
        print(f"HTTP request failed: {e}")
        raise
    except Exception as e:
        print(f"Training failed: {e}")
        raise


async def train_ml_with_data_async(
    case: str = "No_name",
    data_path: str = "automl/data/data_4j1r.csv",  # path to client data folder
    feature_column: List[str] = ["smiles"],
    target_column: List[str] = ["Docking score"],  # All propreties from dataframe you want to calculate in the end,
    regression_props: List[str] = ["Docking score"],
    classification_props: List[str] = [],
    description: str = "",
    timeout: int = 5,
) -> Union[bool, str]:
    """
    Trains a machine learning model using a provided dataset to enable chemical property prediction.
    
    This function prepares data from a CSV or Excel file and sends it to a remote server 
    for model training. The process runs asynchronously to avoid blocking the main application. 
    The trained model can then be used to predict properties of new chemical compounds.
    
    Args:
        case (str, optional): Name of the model being trained. Defaults to "No_name".
        data_path (str, optional): Path to the CSV or Excel file containing the dataset. 
                                    Defaults to "automl/data/data_4j1r.csv".
        feature_column (list, optional): The name of the column containing the input features (e.g., SMILES strings). 
                                          Defaults to ["Smiles"].
        target_column (list, optional):  List of column names representing the properties to be predicted. 
                                          Defaults to ["Docking score"].
        regression_props (list, optional): Column names containing data for regression tasks.  
                                            Skip if no regression is needed. Defaults to ["Docking score"].
        classification_props (list, optional): Column names containing data for classification tasks.
                                              Skip if no classification is needed. Defaults to [].
        timeout (int, optional): The timeout duration (in minutes) for the request. Defaults to 5.
        description (str): A description of the model or training case. Defaults to "".
    
    Returns:
        Dict[str, Any]: Response from the training server containing training results.
    """

    start_time = time.time()

    data_path_obj = Path(data_path)
    ext = data_path_obj.suffix.lower()

    if ext in ('.csv', '.txt', '.tsv'):
        df = pd.read_csv(data_path_obj).to_dict()
    elif ext in ('.xlsx', '.xls', '.xlsm', '.xlsb'):
        df = pd.read_excel(data_path_obj).to_dict()
    elif ext in ('.parquet', '.parq'):
        df = pd.read_parquet(data_path_obj).to_dict()

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

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{conf['url_pred']}/train_ml", 
                json=params,  # aiohttp automatically serializes to JSON
                timeout=aiohttp.ClientTimeout(total=10800)  # 15 minute timeout for the request
            ) as response:
                
                response.raise_for_status()
                result = await response.json()
                
                print(f"Training started successfully for case: {case}")
                print(f"--- {time.time() - start_time:.2f} seconds ---")
                return result

                
    except aiohttp.ClientError as e:
        print(f"HTTP request failed: {e}")
        raise
    except Exception as e:
        print(f"Training failed: {e}")
        raise




async def ml_dl_training_async(
    case: str,
    path: str,
    feature_column: List[str] = ["canonical_smiles"],
    target_column: List[str] = ["docking_score"],
    regression_props: List[str] = ["docking_score"],
    classification_props: List[str] = [],
    poll_interval: int = 30,  # seconds
    max_wait_time: int = 18000,  # 5 hour max
    description: str = None
):
    """
    Trains machine learning and deep learning models for a given case.
    
    This method initiates the training of machine learning (ML) and generative (Gen) models
    using provided data, and actively monitors their progress by polling a server for completion status.
    This ensures models are fully trained before proceeding, enabling tasks such as property prediction
    and molecule generation based on the data.
    
    Args:
        case (str): The identifier for the case to train the models for.
        path (str): The path to the data used for training.
        feature_column (list[str], optional): The name(s) of the column(s) to use as features. Defaults to ["canonical_smiles"].
        target_column (list[str], optional): The name(s) of the column(s) to use as targets. Defaults to ["docking_score"].
        regression_props (list[str], optional): The name(s) of the regression properties. Defaults to ["docking_score"].
        classification_props (list[str], optional): The name(s) of the classification properties. Defaults to [].
    
    Returns:
        None
    """
    start_time = time.time()

    results = {
        "case": case,
        "ml_training": None,
        "gen_training": None,
        "total_time": 0,
        "status": "completed"
    }

    print(f"Starting ML/DL training pipeline for case: {case}")

    try:
        print(f"Starting ML training...")
        ml_task = asyncio.create_task(train_ml_with_data_async(case=case,
                                                    data_path=path,
                                                    feature_column=feature_column,
                                                    target_column=target_column,
                                                    regression_props=regression_props,
                                                    classification_props=classification_props,
                                                    description=description))
        ml_monitor_task = asyncio.create_task(wait_for_training_completion(
                                                case=case,
                                                model_type="pred",
                                                poll_interval=poll_interval,
                                                max_wait_time=max_wait_time,
                                                start_time=start_time
                                            ))
        print(f"Waiting for ML tasks...")

        # Wait for both ML tasks to complete
        ml_result, ml_status = await asyncio.gather(ml_task, ml_monitor_task)
        results["ml_training"] = ml_result
        results["ml_status"] = ml_status
        print(f"ML tasks completed")

        # Then start generative training AND monitoring concurrently  
        gen_task = asyncio.create_task(train_gen_with_data_async(
                                        case=case,
                                        data_path=path,
                                        feature_column=feature_column,
                                        target_column=target_column,
                                        regression_props=regression_props,
                                        classification_props=classification_props,
                                        description=description
                                    ))
        gen_monitor_task = asyncio.create_task(wait_for_training_completion(
                                            case=case,
                                            model_type="gen", 
                                            poll_interval=poll_interval,
                                            max_wait_time=max_wait_time,
                                            start_time=start_time
                                        ))
        
        # Wait for both gen tasks to complete
        gen_result, gen_status = await asyncio.gather(gen_task, gen_monitor_task)
        results["gen_training"] = gen_result
        results["gen_status"] = gen_status

        results["total_time"] = time.time() - start_time


        print(f"Training completed successfully for case: {case}")
        print(f"Total training time: {results['total_time']:.2f} seconds")
        
        #return results

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        results["total_time"] = time.time() - start_time
        print(f"Training failed for case {case}: {e}")
        raise


async def wait_for_training_completion(
    case: str,
    model_type: str,
    poll_interval: int = 30,
    max_wait_time: int = 3600,
    start_time: float = None
) -> Dict[str, Any]:
    """
    Asynchronously poll server for training completion status.
    
    Args:
        case: Case identifier
        model_type: Type of model ("pred" for ML, "gen" for generative)
        poll_interval: How often to check status (seconds)
        max_wait_time: Maximum time to wait (seconds)
        start_time: When training started
        
    Returns:
        Final training status from server
    """
    if start_time is None:
        start_time = time.time()

    model_type_mapper = {'gen': 'generative_models', 
                        'pred': 'ml_models'}

    last_print_time = start_time
    print_interval = 60  # Print status every 60 seconds
    while time.time() - start_time < max_wait_time:
        try:
            # Get current status from server
            status = get_state_from_server(model_type, case)

            if isinstance(status, dict):
                # Check if training is complete
                if status.get(model_type_mapper[model_type], {}).get('status') in {"Trained"}:
                    print(f"{model_type.upper()} training completed for case: {case}")
                    return status
                
                # Check if training in process TODO: on server side implement proper failure mechanism
                # if status.get(model_type_mapper[model_type], {}).get('status') in {"Not Trained", None}:
                #     error_msg = status.get("error", "Unknown error")
                #     raise ValueError(f"{model_type.upper()} training failed: {error_msg}")

            # Print progress periodically
            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                elapsed = current_time - start_time
                print(f"{model_type.upper()} training in progress for case: {case} "
                      f"({elapsed:.0f}s elapsed)")
                last_print_time = current_time
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
            
        except Exception as e:
            print(f"Error checking {model_type} status for {case}: {e}")
            await asyncio.sleep(poll_interval)  # Wait before retry

    raise TimeoutError(
        f"{model_type.upper()} training timeout for case {case} "
        f"after {max_wait_time} seconds"
    )


async def agenerate_with_gan(num: int = 100, timeout: int = 30, case: str = "Alzheimer") -> Tuple[aiohttp.ClientResponse, Dict[str, Any]]:
    params = {'case_': case, 'numb_mol': num}
    url = conf['url_gen'] + '/gan_case_generator'

    timeout_config = aiohttp.ClientTimeout(total=timeout)

    try:
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.post(url, json=params) as resp:
                if resp.status == 200:
                    json_data = await resp.json()
                    if isinstance(json_data, dict):
                        return json_data
                    else:
                        return json.loads(json_data)
                else:
                    error_text = await resp.text()
                    raise Exception(f"API returned status {resp.status}: {error_text}")
    except asyncio.TimeoutError:
        raise Exception(f"Request timed out after {timeout} seconds")
    except aiohttp.ClientError as e:
        raise Exception(f"Network error: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON response: {e}")

           

async def agenerate_with_gan_cyclic(num: int = 10,
    properties_conditions: Optional[Dict[str, str]] = None,
    num_tries: int = 5,
    maximum_error: float = 0.1,
    case: str = 'Alzheimer') -> List[str]:

    tasks = [agenerate_with_gan(num = num, case=case) for _ in range(num_tries)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    available_props = set(results[0].keys()) - {'Smiles'}

    def parse_props(cond_str):
        s = str(cond_str).strip()
        for op in ("<=", ">=", "==", "<", ">"):
            if op in s:
                return op, float(s.split(op)[1].strip())
        raise ValueError(f"Unsupported condition format: {cond_str}")

    required_props = {}
    if properties_conditions is not None:
        required_props = {key: parse_props(properties_conditions[key]) for key in properties_conditions} #{prop: (op, threshold)}

    all_smiles = []
    all_props = defaultdict(list)

    def evaluator(value:float, op_str: str, threshold: float, tolearance:float = 0.1):
        op_map = {
            '<': operator.lt,
            '>': operator.gt,
            '<=': operator.le,
            '>=': operator.ge,
            '==': lambda a, b: abs(a - b) < tolerance
        }
    
        if op_str not in op_map:
            raise ValueError(f"Unsupported operator: {op_str}")
        
        return op_map[op_str](value, threshold)

    check_dict = {key: partial(evaluator, op_str=op, threshold=threshold) for key, (op, threshold) in required_props.items()}

    for result in results:  
        if isinstance(result, Exception):
            continue
        all_smiles.extend(result.get('Smiles', []))
        for prop in available_props:
            all_props[prop].extend(result.get(prop, []))

    def valid(idx, all_props):
        for key in check_dict:
            if key not in available_props:
                continue
            elif not check_dict[key](all_props[key][idx]):
                return False
        return True

    output = []

    safe_props = defaultdict(lambda: ['N/A'] * len(all_smiles))
    safe_props.update(all_props)

    for idx, smile in enumerate(all_smiles):
        if valid(idx, all_props):
            output.append((smile, {key: safe_props[key][idx] for key in check_dict}))
    
    return output[:num] if len(output)>num else output

@tool
def get_state_from_server(url: str = "pred", case: Optional[str] = None) -> Union[dict, str]:
    """
    Retrieves the current status and information about available models from the server.
    This method checks the status of models – whether they are in a training state, have been trained, or are pending – and provides details about their descriptions and associated metrics. It facilitates monitoring the readiness and state of different model deployments. In case of a server error, it returns an appropriate error message.
    If case is provided retrieves the current status and metrics for a specific case/model from the server.
    
    Args:
        url (str): Specifies the server endpoint. Use 'pred' for predictive models or 'gen' for generative models. Defaults to "pred".
        case (str, optional): The name of the case/model to query. Defaults to None
    Returns:
        dict or str: A dictionary containing model state information (status, description, metrics) if successful.  Returns a string  if a server-side error (status code 500) occurs.
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

    if case:
        return state['state'].get(case, f"Case: {case} not found")
    return state["state"]


@tool
def predict_prop_by_smiles(
    smiles_list: List[str], case: str = "no_name_case", timeout: int = 20
) -> dict:
    """
    Predicts molecular properties using pre-trained machine learning models. 
    
    Args:
        smiles_list (List[str]): A list of molecules represented in SMILES format for property prediction.
        case (str, optional): Specifies the model to use for prediction. Defaults to "no_name_case".  Available models can be retrieved using 'get_state_from_server'.
        timeout (int, optional): Sets a time limit (in minutes) for the prediction request. Defaults to 20.
    
    Returns:
        dict: Parsed JSON data representing the predicted properties.
    """
    url = conf["url_pred"] + "/predict_ml"
    params = {"case": case, "smiles_list": smiles_list, "timeout": timeout}
    resp = requests.post(url, json.dumps(params))
    return resp.json()

@tool
def generate_mol_by_case(
    case: str = "Alzheimer",
    n_samples: int = 10,
) -> dict:
    """
    Generates molecules based on a specified model and quantity.
    
    This method facilitates the creation of new molecular structures using pre-trained generative models. It sends a request to a remote server to perform the generation and returns the results.
    
    Args:
        case (str, optional): The name of the generative model to use.  Available model names can be retrieved using `get_state_from_server`. Defaults to "Alzheimer".
        n_samples (int, optional): The number of molecules to generate. Defaults to 10.
    
    Returns:
        dict: A dictionary containing the generated molecules.
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
def generate_mols(
    num: int = 10,
    properties_conditions: Optional[Dict[str, str]] = None,
    num_tries: int = 5,
    case: str = "Alzheimer"
) -> List[str]:
    """
    Generates a set of molecular SMILES strings applying property-based filtering.

    Args:
        num (int, optional): The number of valid SMILES strings to return. Defaults to 10.
        properties_conditions (Optional[Dict[str, str]], optional): 
            A dictionary specifying property-based selection criteria.
            Each entry should map a property name to a condition string 
            (e.g., {"logP": ">=2.5", "QED": "<0.8", "Brenk": "==1.0"}).
            Available properties can be retrieved using 'get_state_from_server'
        num_tries (int, optional): 
            The number of independent generation attempts to perform in parallel. Defaults to 5.
        case (str, optional): The name of the generative model to use.  Available model names can be retrieved using `get_state_from_server`. Defaults to "Alzheimer".

    Returns:
        List[str]: 
            A list of tuples containing the selected SMILES strings and their corresponding 
            evaluated properties that satisfy all specified conditions.
    """
    return asyncio.run(agenerate_with_gan_cyclic(
        num, 
        case=case,
        properties_conditions=properties_conditions, 
        num_tries=num_tries
    ))

@tool
def run_ml_dl_training_by_daemon(
    case: str,
    path: str,
    feature_column: list = ["smiles"],
    target_column: list[str] = ["docking_score"],
    regression_props: list[str] = ["docking_score"],
    classification_props: list = [],
    description: str = None
) -> Union[bool, str]:
    """
    Initiates a machine learning and deep learning training process in the background, utilizing the provided dataset and configuration. 
    This process prepares models for predicting chemical properties or classifying molecules based on input features as well as generating new molecules targeted by this features
    
    Args:
        case (str): A unique identifier for the training case, used to track the process.
        path (str): The file path to the dataset, which should be a CSV, TSV,  Excel or Parquet file.
        feature_column (list, optional):  A list of column names representing the input features for the model. Defaults to ["smiles"].
        target_column (list, optional): A list of column names specifying the properties to be predicted or classified. This list must not be empty. Defaults to ["docking_score"].
        regression_props (list, optional): A list of column names used for regression tasks. This list should generally align with the feature columns. Defaults to ["docking_score"].
        classification_props (list, optional): A list of column names used for classification tasks. Use an empty list if classification is not required. Defaults to [].
        description (str, optional): Description of the data and case
    Returns:
        None
    
    Raises:
        ValueError: If target_column or feature_column are empty, or if specified columns are not found in the dataset, or if the dataset is too small (less than 300 samples).
        FileNotFoundError: If the specified data file does not exist.
    
    Note: At least one of regression_props or classification_props must be specified for the training to proceed.
    """
    if isinstance(feature_column, str):
        feature_column = [feature_column]
    if isinstance(target_column, str):
        target_column = [target_column]

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

    data_path_obj = Path(path)
    ext = data_path_obj.suffix.lower()

    if ext in ('.csv', '.txt', '.tsv'):
        df = pd.read_csv(path)
    elif ext in ('.xlsx', '.xls', '.xlsm', '.xlsb'):
        df = pd.read_excel(data_path_obj)
    elif ext in ('.parquet', '.parq'):
        df = pd.read_parquet(data_path_obj)

    # if len(df) < 300:
    #     raise ValueError(
    #         "Training on this data is impossible. The dataset is too small!"
    #     )

    df_columns = set(df.columns.tolist())
    for column in feature_column:
        if column not in df_columns:
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )
    for column in target_column:
        if column not in df_columns:
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )

    # delete molecules with len more 200
    # clear_df = filter_valid_strings(df, feature_column[0])
    # clear_df.to_csv(path)

    async def _run_training():
        try:
            results = await ml_dl_training_async(case=case, 
                                path=path, 
                                feature_column=feature_column,
                                target_column=target_column,
                                regression_props=regression_props,
                                classification_props=classification_props,
                                description=description)
            print(f"Background training completed for {case}")
            return results
        except Exception as e:
            print(f"Background training failed for {case}: {e}")
            return f"Background training completed for {case}"

    def _run_in_background():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run_training())
        loop.close()

    try:
        # If we're already in an async context, use create_task
        loop = asyncio.get_running_loop()
        task = loop.create_task(_run_training())
        print(f"Async training daemon started for case: {case}")
        return f"Async training daemon started for case: {case}"
    except RuntimeError:
        # If no event loop running, create new one in background thread

        import threading
        thread = threading.Thread(target=_run_in_background, daemon=False)
        thread.start()
        print(f"🎯 Thread-based training daemon started for case: {case}")
        return f"Thread-based training daemon started for case: {case}"
    

agents_tools = [
    run_ml_dl_training_by_daemon,
    get_state_from_server,
    generate_mols,
    # generate_mol_by_case,
    predict_prop_by_smiles,
]

