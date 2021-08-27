# %%
import ntpath
import subprocess
import tempfile
from typing import List, Set, Union

import sentencepiece as spm
from transformers import MarianMTModel, MarianTokenizer

from med_dict import med_dict
from process_free_text import process_free_text
from stopwords_ptbr import stopwords_ptbr

# %% OPENMT Felipe Soares et al.
tokenizer_model = "models/espt_enSP32k.model" #https://zenodo.org/record/3346802/files/espt_enSP32k.model?download=1

translation_model = "models/espt_en_model.pt" #https://zenodo.org/record/3346802/files/espt_en_model.pt?download=1

sp = spm.SentencePieceProcessor()
sp.load(tokenizer_model)

# %% Transformers module
marian_model_name = "Helsinki-NLP/opus-mt-roa-en"
marian_tokenizer = MarianTokenizer.from_pretrained(marian_model_name)
marian_model = MarianMTModel.from_pretrained(marian_model_name)

# %%

def opennmt_translate(text: str, python_path: str, spm_decode_path: str) -> str:
    """Return translated text (Brazilian Portuguese to English) by Felipe Soares' especialized medical translator (source: https://github.com/PlanTL-SANIDAD/Medical-Translator-WMT19/blob/master/tokenize_SP.sh).

    Args:
        text (str): Medical text in Brazilian Portuguese
        python_path (str): Python interpreter path (useful when Python is installed in a virtual environment, such as conda)
        spm_decode_path (str): Sentencepiece decoder path (useful when  installed in a virtual environment, such as conda)

    Returns:
        str: Translated medical text (in English)
    """

    processed_text=process_free_text(text, dictionary=med_dict, stopwords=stopwords_ptbr)

    tokenized_tempfile = tempfile.NamedTemporaryFile(mode="w")
    translated_tempfile = tempfile.NamedTemporaryFile(mode="w")
    detokenized_tempfile = tempfile.NamedTemporaryFile(mode="w")

    with open(tokenized_tempfile.name, "w") as f:
        f.write(f"__opt_src_pt __opt_tgt_en {' '.join(sp.encode_as_pieces(processed_text))}\n")

    subprocess.run(
        [
            python_path,
            "translate.py",
            "-model",
            translation_model,
            "-src",
            tokenized_tempfile.name,
            "-replace_unk",
            "-output",
            translated_tempfile.name,
        ]
    )

    subprocess.run(
        [
            spm_decode_path,
            "--model",
            tokenizer_model,
            "--input",
            translated_tempfile.name,
            "--output",
            detokenized_tempfile.name,
        ]
    )

    with open(detokenized_tempfile.name, "r") as f:
        translated_text = f.read().strip()

    tokenized_tempfile.close()
    translated_tempfile.close()
    detokenized_tempfile.close()

    return translated_text


# %%
def marian_translate(text: str) -> str:
    """Returns translated text (Brazilian Portuguese to English) with generalist translator.

    Args:
        text (str): Medical text in Brazilian Portuguese

    Returns:
        str: Translated medical text (in English)
    """

    processed_text=process_free_text(text, dictionary=med_dict, stopwords=stopwords_ptbr)

    translated = marian_model.generate(
        **marian_tokenizer.prepare_seq2seq_batch(processed_text, return_tensors="pt")
    )
    translated_text = [
        marian_tokenizer.decode(t, skip_special_tokens=True) for t in translated
    ]

    return translated_text[0]

# %%
def create_phenotagger_input(patient_id: Union[int, str], translated_text: str) -> str:
    """Format medical translated text into PubTator input format for Ling Luo's PhenoTagger.

    Args:
        patient_id (Union[int, str]): Patient unique id
        translated_text (str): Medical text (in English). The output of opennmt_translate() or marian_translate() functions

    Returns:
        str: Multi-line string in PubTator format input for PhenoTagger
    """    
    
    pubtator_input = f"""{patient_id}|t|patient_id: {patient_id}
{patient_id}|a|{translated_text}

"""

    return pubtator_input

# %%
def run_phenotagger(pubtator_input: str, python_path: str, tagger_path: str) -> List[str]:
    """Execute Luo Ling's PhenoTagger against English medical text in PubTator format. Source: https://github.com/ncbi-nlp/PhenoTagger.

    Args:
        pubtator_input (str): Multi-line string. The output of create_phenotagger_input() function
        python_path (str): Python interpreter path (useful when Python is installed in a virtual environment, such as conda)
        tagger_path (str): Full path of PhenoTagger_tagging.py Python script (ends with /PhenoTagger/src/). It only works if the current workind directory (cwd) is this folder.

    Returns:
        List[str]: List of HPO ids (strings) extracted from the medical text in English.
    """    

    input_tempdir = tempfile.TemporaryDirectory()

    output_tempdir = tempfile.TemporaryDirectory()

    pubtator_input_tempfile = tempfile.NamedTemporaryFile(mode="w", dir=input_tempdir.name, suffix="_input.PubTator", delete=False)

    with open(pubtator_input_tempfile.name, "w") as f:
        f.write(pubtator_input)
    
    subprocess.run([
        python_path,
        "PhenoTagger_tagging.py",
        "-i",
        f"{input_tempdir.name}/",
        "-o",
        f"{output_tempdir.name}/"
    ], cwd=tagger_path)
    
    with open(f"{output_tempdir.name}/{ntpath.basename(pubtator_input_tempfile.name)}", "r") as f:

        hpos = []
        for line in f:
            for word in line.strip().split("\t"):
                if word.startswith("HP:"):
                    hpos.append(word)

    input_tempdir.cleanup()
    output_tempdir.cleanup()
        
    return hpos

# %%
def hpo_extractor(patient_id: Union[int, str], text: str, python_path: str, spm_decode_path: str, tagger_path: str) -> str:
    """Wrapper function to translate medical text in Brazilian Portuguese into English and predict HPO ids from it with Ling Luo's PhenoTagger.

    Args:
        patient_id (Union[int, str]): Patient unique id
        text (str): Medical text in Brazilian Portuguese
        python_path (str): Python interpreter path (useful when Python is installed in a virtual environment, such as conda)
        spm_decode_path (str): Sentencepiece decoder path (useful when  installed in a virtual environment, such as conda)
        tagger_path (str): Full path of PhenoTagger_tagging.py Python script (ends with /PhenoTagger/src/). It only works if the current workind directory (cwd) is this folder.

    Returns:
        str: Set of HPO ids (strings) extracted from the medical text in English.
    """    

    translated_opennmt = opennmt_translate(text, python_path, spm_decode_path)

    translated_marian = marian_translate(text)

    pubtator_opennmet = create_phenotagger_input(patient_id, translated_opennmt)

    pubtator_marian = create_phenotagger_input(patient_id, translated_marian)

    pubtator_input = pubtator_opennmet + pubtator_marian

    hpos = run_phenotagger(pubtator_input, python_path, tagger_path)

    hpos_set = set(hpos)

    return ";".join(hpos_set)
