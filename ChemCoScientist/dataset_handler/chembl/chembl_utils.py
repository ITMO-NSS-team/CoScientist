import pandas as pd
from langchain_core.tools import tool


class ChemblLoader:
    """
    A class for accessing and managing chemical compound data from the ChEMBL database. It provides functionality to load data, apply filters based on various criteria (e.g., activity values, molecule properties), and prepare the data for downstream analysis. This class facilitates efficient retrieval of relevant chemical information for use in research and data exploration.
    
        Attributes:
            columns (dict): A dictionary mapping column names to their data types.
            data_path (str): The path to the CSV file.
            data (pd.DataFrame or None): The loaded data, or None if not loaded.
    """


    def __init__(self, load: bool = False, file_path: str = None):
        """
        Initializes the ChemblLoader with optional data loading.
        
        Args:
            load (bool): If True, the data will be loaded immediately upon initialization.
            file_path (str): The path to the CSV file containing the ChEMBL data.
        
        Returns:
            None
        
        WHY: This method prepares the loader to handle ChEMBL data, defining the expected columns and optionally loading the data from a specified file. Defining structure of the data allows to consistently process and analyze chemical compounds and their properties. Loading the data immediately makes it readily available for downstream tasks, while the option to defer loading provides flexibility in memory management and data handling.
        """
        self.columns = chembl_columns = {
            "ChEMBL ID": str,
            "Name": str,
            "Synonyms": str,
            "Type": str,
            "Max Phase": int,
            "Molecular Weight": float,
            "Targets": int,
            "Bioactivities": int,
            "AlogP": float,
            "Polar Surface Area": float,
            "HBA": int,
            "HBD": int,
            "#RO5 Violations": int,
            "#Rotatable Bonds": int,
            "Passes Ro3": object,
            "QED Weighted": float,
            "CX Acidic pKa": float,
            "CX Basic pKa": float,
            "CX LogP": float,
            "CX LogD": float,
            "Aromatic Rings": int,
            "Structure Type": str,
            "Inorganic Flag": int,
            "Heavy Atoms": int,
            "HBA (Lipinski)": int,
            "HBD (Lipinski)": int,
            "#RO5 Violations (Lipinski)": int,
            "Molecular Weight (Monoisotopic)": float,
            "Np Likeness Score": float,
            "Molecular Species": str,
            "Molecular Formula": str,
            "Smiles": str,
            "Inchi Key": str,
            "Inchi": str,
            "Withdrawn Flag": bool,
            "Orphan": int,
            "Records Key": object,
            "Records Name": object,
        }
        self.data_path = file_path
        if load and file_path:
            self.data = self._load_data()
        else:
            self.data = None

    def _load_data(self):
        """
        Loads data from a CSV file into a Pandas DataFrame.
        
        Args:
            None
        
        Returns:
            None
        """
        self.data = pd.read_csv(
            self.data_path, delimiter=";", on_bad_lines="skip", engine="python"
        )

    def get_columns(self) -> dict:
        """
        Returns the columns dictionary used for data processing.
        
        Args:
            None
        
        Returns:
            dict: A dictionary mapping column names to their corresponding indices or properties within the dataset. This dictionary is used to identify and access specific data points in the chemical data.
        """
        return self.columns

    def get_filtered_data(
        self, selected_columns: list, filters: dict = None
    ) -> pd.DataFrame:
        """
        Filters the chemical compound data to return a focused subset based on specified criteria.
        
        Args:
            selected_columns (list): A list of column names to include in the resulting DataFrame.  If 'Smiles' is not included, it is automatically added.
            filters (dict, optional): A dictionary specifying filter conditions. Keys are column names and values are:
                - A tuple (min_val, max_val) for numerical range filtering (inclusive).
                - A string or boolean for exact match filtering.
        
        Returns:
            pd.DataFrame: A DataFrame containing the filtered chemical compound data.
        """
        if "Smiles" not in selected_columns:
            selected_columns.append("Smiles")

        if not self.data:
            self._load_data()

        selected_columns = [i for i in selected_columns if i in self.columns.keys()]

        df = self.data[selected_columns]

        if filters:
            for column, condition in filters.items():
                if column in df.columns:
                    if isinstance(condition, tuple) and len(condition) == 2:
                        min_val, max_val = condition
                        df = df[(df[column] >= min_val) & (df[column] <= max_val)]
                    elif isinstance(condition, (str, bool)):
                        df = df[df[column] == condition]

        return df


@tool
def get_filtered_data(
    self, selected_columns: list, filters: dict = None
) -> pd.DataFrame:
    """
    Filters a DataFrame to include only desired columns and rows matching specified criteria.
    
    Args:
        selected_columns (list): A list of column names to include in the filtered DataFrame.  "Smiles" column will be added automatically if it's not in the list.
        filters (dict, optional): A dictionary where keys are column names and values are filter conditions. 
                                  Conditions can be:
                                  - A tuple (min_val, max_val) for numerical range filtering (inclusive).
                                  - A string or boolean for exact value matching.
    
    Returns:
        pd.DataFrame: A new DataFrame containing only the selected columns and rows that satisfy the filter conditions.
    """
    if "Smiles" not in selected_columns:
        selected_columns.append("Smiles")

    if not self.data:
        self._load_data()
    df = self.data[selected_columns]

    if filters:
        for column, condition in filters.items():
            if column in df.columns:
                if isinstance(condition, tuple) and len(condition) == 2:
                    min_val, max_val = condition
                    df = df[(df[column] >= min_val) & (df[column] <= max_val)]
                elif isinstance(condition, (str, bool)):
                    df = df[df[column] == condition]

    return df


if __name__ == "__main__":
    file_path = "./ChemCoScientist/dataset_handler/chembl/p1_short.csv"
    selected_columns = ["Molecular Weight"]
    filters = {
        "Molecular Weight": (150, 500),
    }
    client = ChemblLoader(True, file_path)
    df = client.get_filtered_data(selected_columns, filters)
    print(df)
