import nltk
from nltk.corpus import stopwords
# nltk.download("stopwords")

stopwords_ptbr = stopwords.words("portuguese")

more_stopwords = [
    "inclusão",
    "critério",
    "critérios",
    "presença",
    "seguinte",
    "seguintes",
    "idade",
    "anos",
    "preenchido",
    "preenchidos",
    "eou"
]

for word in more_stopwords:
    stopwords_ptbr.append(word)