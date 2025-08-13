import os
import sys
from typing import List

sys.path.append(os.getcwd())
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from infrastructure.automl.utils.base_state import *
from fedot.api.main import Fedot,Pipeline
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
import itertools

def input_data_preparing(case:str,
                         problem:str = 'regression',
                         split:bool = True):
    """
    Prepares molecular data for machine learning tasks.
    
    Reads data from a CSV file, converts SMILES strings into numerical molecular representations
    (Morgan fingerprints), and formats the data into a standardized structure suitable for model training
    or prediction. This process ensures consistent and effective inputs for downstream analysis.
    
    Args:
        case (str): Specifies the dataset case to use, determining the data source and features.
        problem (str, optional): The machine learning problem type (e.g., 'regression'). Defaults to 'regression'.
        split (bool, optional):  Indicates whether to split the data into training and testing sets. Defaults to True.
    
    Returns:
        tuple[InputData, InputData] | InputData: If `split` is True, returns a tuple containing the training and testing `InputData` objects.
                                                 Otherwise, returns a single `InputData` object containing the entire dataset.
    """
    state = TrainState()
    df = pd.read_csv(state(case,'ml')['data_path']).iloc[:1000,:]
    if type(state(case,'ml')['feature_column'])==str:
        df_x = df[state(case,'ml')['feature_column']].apply(lambda x: Chem.MolFromSmiles(x))
    elif type(state(case,'ml')['feature_column'])==list:
        df_x = df[state(case,'ml')['feature_column'][0]].apply(lambda x: Chem.MolFromSmiles(x))
    df_x = df_x.dropna().apply(
        lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048)
    )
    X = np.array(df_x.tolist())
    y = df[state(case,'ml')["Predictable properties"][problem]]
    X = pd.DataFrame(data=X)
    y = pd.DataFrame(data=y)
    data = InputData.from_dataframe(X,
                                y,task=problem)
    data.supplementary_data.is_auto_preprocessed = True
    if split:
        train, test = train_test_data_setup(data)

        return  train, test
    else:
        return data
    
def input_data_preparing_from_list(case:str,
                                   data:List[str],
                                    split:bool = True,
                                    problem:str='regression'):
    """
    Prepares molecular data from a list of SMILES strings for machine learning.
    
    This method transforms SMILES strings into a numerical representation 
    suitable for model training and prediction. It converts these strings 
    into molecular fingerprints, which capture essential structural information,
    and organizes the data into an `InputData` object for streamlined processing. 
    
    Args:
        case (str): Specifies the configuration or context for data handling,
            influencing potential data selection or feature engineering.
        data (List[str]): A list of SMILES strings representing chemical structures.
        split (bool, optional):  If True, the data is split into training and 
            testing sets. Defaults to True.
        problem (str, optional): Specifies the type of machine learning problem 
            (e.g., 'regression', 'classification'). Defaults to 'regression'.
    
    Returns:
        tuple or InputData: If `split` is True, returns a tuple containing the 
            training and testing `InputData` objects. Otherwise, returns the 
            combined `InputData` object.
    """
    state = TrainState()
    data = {"Smiles":data}
    df = pd.DataFrame(data=data)
    df_x = df['Smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df_x = df_x.dropna().apply(
        lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048)
    )
    X = np.array(df_x.tolist())
    #y = df[state(case,'ml')['data']['target_column']]
    X = pd.DataFrame(data=X)
    y = pd.DataFrame(data={})
    data = InputData.from_dataframe(X,y,
                                task=problem)
    data.supplementary_data.is_auto_preprocessed = True
    if split:
        train, test = train_test_data_setup(data)

        return  train, test
    else:
        return data

def run_train_automl(case:str,
                     timeout:int = 5,
                     path_to_save = r'generative_models_data/generative_models/transformer_auto'):
    """
    Runs automated machine learning training for a given case and its predictable properties.
    
    This method prepares data, trains a Fedot model for each predictable property 
    (regression or classification), saves the resulting pipeline, and calculates performance metrics. 
    This process helps establish a baseline model for each property within the specified case.
    
    Args:
        case (str): The identifier for the case being processed.  This links the training process to specific data and configuration.
        timeout (int, optional): The maximum time in minutes allowed for model optimization. Defaults to 5.
        path_to_save (str, optional): The base path to save the trained models. Defaults to r'generative_models_data/generative_models/transformer_auto'.
    
    Returns:
        None
    """
    state = TrainState()
    metrics = {'regression':None,'classification':None}
    state.ml_model_upd_status(case=case,status=1)
    for problem in state(case,'ml')["Predictable properties"]:
        train, test = input_data_preparing(case=case, problem = problem)
        if problem == 'regression':
            available_secondary_operations = ['catboostreg','rfr', 'xgboostreg']
        elif problem == 'classification':
            available_secondary_operations = ['catboost','rf', 'xgboost']
        if not os.path.isdir(path_to_save+f'_{case}'+f'_{problem}'):
            os.mkdir(path_to_save+f'_{case}'+f'_{problem}')
        state.ml_model_upd_status(case=case,model_weight_path=path_to_save+f'_{case}'+f'_{problem}',problem=problem)
        model = Fedot(
            problem=problem,
            preset='fast',  # Options: 'fast', 'stable', 'best_quality', etc.
            timeout=timeout,  # Minutes for optimization
            with_tuning=True,  # Allow tuning mode
            n_jobs=-1,  # CPU cores to use (-1 = all)
            cv_folds=5,  # Cross-validation folds
        available_operations = available_secondary_operations,

        )
        
        model.fit(features=train.features, target=train.target)
        model.current_pipeline.save(path=path_to_save+f'_{case}'+f'_{problem}', create_subdir=False, is_datetime_in_path=False)
        

        model.predict(features=test.features)
        metrics[problem] = model.get_metrics(test.target)
    state.ml_model_upd_status(case=case,metric=metrics,status=2)
        

#Test function
# def run_predict_automl_from_data(case:str,
#                      timeout:int = 5,
#                      path_to_save = r'generative_models_data/generative_models/transformer_auto'):
#     state = TrainState()
#     data = input_data_preparing(case=case, task = state(case,'ml')['problem'], split=False)
#     pipeline = Pipeline().load(state(case,'ml')['weights_path'])
#     #model.current_pipeline.save(path=path_to_save+f'_{case}', create_subdir=False, is_datetime_in_path=False)
#     resutls = pipeline.predict(input_data=data)
#     print(resutls)

def run_predict_automl_from_list(case:str,
                     data:List[str]):
    """
    Runs AutoML prediction on a list of data points for specified properties.
    
    This method prepares data, loads pre-trained machine learning pipelines tailored to a given case,
    and generates predictions for both calculable and predictable properties. It orchestrates the
    prediction process by utilizing pre-defined target columns and identifying appropriate models.
    
    Args:
        case (str): A key identifying the specific modeling context or experimental setup.
        data (List[str]): A list of data points (represented as strings) for which predictions are needed.
    
    Returns:
        dict: A dictionary where keys are property names (e.g., 'docking_score', 'QED') and
              values are lists of corresponding predictions for each input data point.
    """
    state = TrainState()
    properties = {}
    state.ml_model_upd_data(case=case,
                            target_column=['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50'],
                           )
    predicteble_props = [state(case,'ml')['Predictable properties'][i] for i in state(case,'ml')['Predictable properties']]
    predicteble_props = list(itertools.chain(*predicteble_props))
    calc_props = [i for i in state(case,'ml')['target_column'] if i not in predicteble_props]
    for p in calc_props:
        properties[p] = state()['Calculateble properties'][p](data)

    for problem in state(case,'ml')["Predictable properties"]:

        data_preapred = input_data_preparing_from_list(case=case, split=False,data=data,problem=problem)
        pipeline = Pipeline().load(state(case,'ml')['weights_path'][problem])
        resutls = pipeline.predict(input_data=data_preapred)
        for i,prop in enumerate(state(case,'ml')["Predictable properties"][problem]):
            if len(resutls.predict.shape)>1:
                properties[prop] = list(resutls.predict[:,i])
            else:
                properties[prop] = list(resutls.predict)

    return properties



if __name__=='__main__':

#####
#Example for train
    # state = TrainState()

    # data_path = "automl\data\data_4j1r.csv"
    # task = 'Brain_cancer_test'
    # feature_column=['canonical_smiles']
    # target_column=['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50']
    # regression_props = ['LogP','docking_score',"Synthetic Accessibility",'QED']
    # classification_props = ['IC50']

    # state.add_new_case(task,rewrite=True)
    # state.ml_model_upd_data(case=task,
    #                         data_path = data_path,
    #                         feature_column=feature_column,
    #                         target_column=target_column,
    #                         predictable_properties={"regression":regression_props, "classification":classification_props})
    
    # run_train_automl(case=task,path_to_save='automl/trained_data')
#######


######
#Example for inference
    task = "Brain_cancer_test"
    state = TrainState()
    smiles_list = pd.read_csv(state(task,'ml')['data_path']).iloc[:10,:]['canonical_smiles'].to_list()
    print(run_predict_automl_from_list(case=task,data=smiles_list))
