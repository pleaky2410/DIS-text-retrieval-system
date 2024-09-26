import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
from enum import Enum
import pickle
import json
import os
from concurrent.futures import ProcessPoolExecutor

CORPUS_PATH = "data/corpus.json"
DEV_PATH = "data/dev.csv"
TEST_PATH = "data/test.csv"
TRAIN_PATH = "data/train.csv"



LOAD_BATCH_SIZE = 10000

nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class Lang(Enum):
    ENGLISH = 'en'
    FRENCH = 'fr'
    GERMAN = 'de'
    SPANISH = 'es'
    ITALIAN = 'it'
    ARABIC = 'ar'
    KOREAN = 'ko'


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


def compute_vocabulary(documents, save_path="./.cache/vocabulary"):
    """
    Compute the vocabulary of a list of documents.
    """
    vocabulary = set()
    for doc in documents:
        vocabulary.update(doc.getTokens())
    
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(vocabulary, f)
    
    return vocabulary

class Document:
    """
    Represents a document in a specific language.
    """
    def __init__(self, content: str, lang: Lang, id: int, preprocess_func):
        self.tokens = preprocess_func(content)
        self.lang = lang
        self.id = id
    
    def stem(self):
        self.tokens = stem_tokens(self.tokens)
    
    def lemmatize(self):
        self.tokens = lemmatize_tokens(self.tokens)
    
    def __str__(self):
        return f"Document {self.id} - {self.lang.name}: {self.tokens[:30]}..."
    
    def getTokens(self):
        return self.tokens
    
    def getLang(self):
        return self.lang
    
    def getId(self):
        return self.id
        

class Query:
    """
    Represents a query in a specific language.
    """

    def __init__(self, content: str, lang: Lang):
        self.content = content
        self.lang = lang
    
    def preprocess(self):
        self.tokens = preprocess_text(self.content, self.lang)

    def stem(self):
        self.tokens = stem_tokens(self.tokens)
    
    def lemmatize(self):
        self.tokens = lemmatize_tokens(self.tokens)

    def __str__(self):
        return f"Query - {self.lang.name}: {self.content[:30]}..."
    
    def tokenize(self):
        if hasattr(self, 'tokens') is False or self.tokens is None:
            self.tokens = self.content.split()
        else:
            raise Exception("Query already tokenized")
    
    def getTokens(self):
        if hasattr(self, 'tokens') is False or self.tokens is None:
            self.tokenize()
        return self.tokens
    
    def getLang(self):
        return self.lang
    
def simple_split(content):
    return content.split()

PREPROCESS_FUNC = simple_split

def batch_load_documents(executor, path=CORPUS_PATH, preprocess_func=simple_split):
    """
    Load documents from a file.
    """

    # Reading the JSON file
    with open(path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        len_corpus = len(corpus)
        print(f"Loaded {len(corpus)} documents.")
        for i in range(0, len(corpus), LOAD_BATCH_SIZE):
            last_idx = min(i + LOAD_BATCH_SIZE, len_corpus)
            batch_content = corpus[i:last_idx]
            batch = list(executor.map(create_doc, batch_content, chunksize=LOAD_BATCH_SIZE // 32))

            yield batch

    # for doc in corpus:
    #     docid = doc["docid"]
    #     text = doc["text"]
    #     lang = doc["lang"]
    #     batch.append(Document(text, Lang(lang), docid, preprocess_func))

    #     if len(batch) == LOAD_BATCH_SIZE:
    #         yield batch
    #         batch = []
        
    # if len(batch) > 0:
    #     yield batch

def save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def create_doc(raw_doc):
    return Document(raw_doc["text"], Lang(raw_doc["lang"]), raw_doc["docid"], simple_split)
