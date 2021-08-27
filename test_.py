# %%
import pandas as pd

# from process_free_text import process_free_text
# from med_dict import med_dict
# from stopwords_ptbr import stopwords_ptbr

from hpo_extractor import hpo_extractor

# %%
PYTHON_PATH="/home/antonio/miniconda3/envs/phenotagger/bin/python"
SPM_DECODE_PATH="/home/antonio/miniconda3/envs/phenotagger/bin/spm_decode"
TAGGER_PATH="/mnt/e/Documents/Pós-doutorado/Genomika Albert Einstein/Projetos Genomika/25 NLP classificação diagnóstico/hpo_extractor/PhenoTagger/src/"

# %%
URL = "https://metabase.genomika.com.br/public/question/06419b00-a519-4484-9f7c-2a04b50b1e92"

df = pd.read_csv(f"{URL}.csv")

## %%
df1 = df.head(5)

df1['hpos_nlp'] = df1.apply(lambda x: hpo_extractor(x["patient_id"],x["orders_redcappayload → survey_content"], PYTHON_PATH, SPM_DECODE_PATH, TAGGER_PATH), axis=1)
