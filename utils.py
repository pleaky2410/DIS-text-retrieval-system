import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
from enum import Enum

nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class Lang(Enum):
    ENGLISH = 1
    FRENCH = 2
    GERMAN = 3
    SPANISH = 4
    ITALIAN = 5
    ARABIC = 6
    KOREAN = 7


def get_stopwords(lang: Lang):
    if lang.value == Lang.ENGLISH.value:
        return set(stopwords.words('english'))
    
    elif lang.value == Lang.FRENCH.value:
        return set(stopwords.words('french'))
    
    elif lang.value == Lang.GERMAN.value:
        return set(stopwords.words('german'))

    elif lang.value == Lang.SPANISH.value:
        return set(stopwords.words('spanish'))

    elif lang.value == Lang.ITALIAN.value:
        return set(stopwords.words('italian'))
    
    elif lang.value == Lang.ARABIC.value:
            return set() # TODO

    elif lang.value == Lang.KOREAN.value:
            return set() # TODO
    
    else:
         raise Exception("Language not supported")


def preprocess_text(text: str, lang: Lang):
    """
    Preprocess the input text by lowercasing, removing punctuation, 
    tokenizing, removing stop words, and stemming.

    Parameters:
    text (str): The input text to preprocess.

    Returns:
    list: A list of processed tokens.
    """
    # Lowercase the text and remove punctuation in a single step
    text = re.sub(r'[^\w\s]', '', text.lower())

    stopwords_lang = get_stopwords(lang)
    
    tokens = [
        token
        for token in text.split()
        if token not in stopwords_lang
    ]

    return tokens


def stem_tokens(tokens):
    """
    Stems the received tokens 
    """
    return [stemmer.stem(token) for token in tokens]


def lemmatize_tokens(tokens):
    """
    Lemmatizes the received tokens
    """
    return [lemmatizer.lemmatize(token) for token in tokens]
    

def cossine_similarity(vec1, vec2):
    """
    Calculate the cossine similarity between vec1 and vec2
    """

    dot_prod = np.dot(vec1, vec2)

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    return dot_prod / (norm_vec1 * norm_vec2)