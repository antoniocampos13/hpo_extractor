from pathlib import Path
import pandas as pd

project_folder = Path().resolve()

medical_dictionary = "med_dict.xlsx"

med_dictdf = pd.read_excel(medical_dictionary, sheet_name="med_dict", engine="openpyxl")

med_dict = {key: value for key, value in zip(med_dictdf["sigla"], med_dictdf["en"])}