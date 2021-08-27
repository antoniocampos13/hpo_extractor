# %%
import re
from string import punctuation

from typing import List

# %%
def strip_punctuation(string: str) -> str:
    """Remove punctuation from string

    Args:
        string (str): String

    Returns:
        str: String without punctuation
    """    
    return "".join(
        character.strip(punctuation)
        for character in string
        if character.strip(punctuation)
    )

def check_abbr(string: str, dictionary: dict) -> str:
    """If abbreviation is in string, return full term

    Args:
        string (str): String
        dictionary (dict): Dictionary of abbreviations and their full meaning

    Returns:
        str: Full meaning of abbreviation
    """    
    if string in dictionary:
        return dictionary[string]
    else:
        return string

def process_free_text(string: str, dictionary: dict, stopwords: List[str]) -> str:
    """Remove punctuation from text, remove stopwords and substitute abbreviations by ther full meaning

    Args:
        string (str): Text
        dictionary (dict): Dictionary of abbreviations and their full meaning
        stopwords (List[str]): List of Brazilian Portuguese common stopwords to be removed

    Returns:
        str: Processed text
    """    
    
    strstrip = strip_punctuation(re.sub("\d+|(\s+[A-Za-z]\.\s+)|\n|( )+|\r", " ", str(string)).strip())
        
    strlist = strstrip.split()

    processed = []

    for word in strlist:
        if len(word) > 1:
            if word.lower() in stopwords:
                continue
            else:
                processed.append(check_abbr(word, dictionary))
        else:
            continue

    return " ".join(filter(None, processed)).lower()
