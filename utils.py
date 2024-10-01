import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
from enum import Enum
import pickle
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import ijson
from functools import partial
from ko_ww_stopwords.stop_words import ko_ww_stop_words
import kr_sentence.tokenizer 
import pyarabic.araby
import pyarabic.araby_const
import tkseem as tk
import pyarabic
import spacy
import stanza
import gc
from nltk.stem import ISRIStemmer


CORPUS_PATH = "data/corpus.json"
DEV_PATH = "data/dev.csv"
TEST_PATH = "data/test.csv"
TRAIN_PATH = "data/train.csv"



LOAD_BATCH_SIZE = 200

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")
nlp_de = spacy.load("de_core_news_sm")
nlp_es = spacy.load("es_core_news_sm")
nlp_it = spacy.load("it_core_news_sm")
nlp_ko = spacy.load("ko_core_news_sm")

class Lang(Enum):
    ENGLISH = 'en'
    FRENCH = 'fr'
    GERMAN = 'de'
    SPANISH = 'es'
    ITALIAN = 'it'
    ARABIC = 'ar'
    KOREAN = 'ko'


def get_nlp_pipeline(lang: Lang):
    global nlp_en, nlp_fr, nlp_de, nlp_es, nlp_it, nlp_ar, nlp_ko
    if lang.value == Lang.ENGLISH.value:
        if nlp_en is None:
            nlp_en = spacy.load("en_core_web_sm")
        return nlp_en
    elif lang.value == Lang.FRENCH.value:
        if nlp_fr is None:
            nlp_fr = spacy.load("fr_core_news_sm")
        return nlp_fr
    elif lang.value == Lang.GERMAN.value:
        if nlp_de is None:
            nlp_de = spacy.load("de_core_news_sm")
        return nlp_de
    elif lang.value == Lang.SPANISH.value:
        if nlp_es is None:
            nlp_es = spacy.load("es_core_news_sm")
        return nlp_es
    elif lang.value == Lang.ITALIAN.value:
        if nlp_it is None:
            nlp_it = spacy.load("it_core_news_sm")
        return nlp_it
    elif lang.value == Lang.KOREAN.value:
        if nlp_ko is None:
            nlp_ko = spacy.load("ko_core_news_sm")
        return nlp_ko
    else:
        raise Exception("Language not supported")

def cleanup_all():
    global nlp_en, nlp_fr, nlp_de, nlp_es, nlp_it, nlp_ko
    nlp_en = None
    nlp_fr = None
    nlp_de = None
    nlp_es = None
    nlp_it = None
    nlp_ko = None
    del nlp_en, nlp_fr, nlp_de, nlp_es, nlp_it, nlp_ko
    gc.collect()

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
            return set(stopwords.words('arabic')) # TODO

    elif lang.value == Lang.KOREAN.value:
            return set(ko_ww_stop_words) # TODO
    
    else:
         raise Exception("Language not supported")

def preprocess_text(text: str, lang: Lang, args):
    """
    Preprocess the input text by lowercasing, removing punctuation, 
    tokenizing, removing stop words, and stemming.

    Parameters:
    text (str): The input text to preprocess.

    Returns:
    list: A list of processed tokens.
    """
    # Lowercase the text and remove punctuation in a single step
    if args.lowercase:
        text = text.lower()
    
    if args.remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    if lang == Lang.ARABIC:
        tokens = nltk.word_tokenize(text)
    
    else :
        nlp = get_nlp_pipeline(lang)
        doc = nlp(text)
        tokens = [token.text for token in doc]

    if args.stopwords:
        stopwords_lang = get_stopwords(lang)
        tokens = [
            token
            for token in tokens
            if token not in stopwords_lang
        ]
    
    if args.lemmatize:
        if lang == Lang.ARABIC:
            stemmer = ISRIStemmer()
            tokens = [stemmer.suf32(word) for word in tokens]
        else:
            tokens = [token.lemma_ for token in doc]
    
    return tokens


def cossine_similarity(vec1, vec2):
    """
    Calculate the cossine similarity between vec1 and vec2
    """

    dot_prod = np.dot(vec1, vec2)

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    return dot_prod / (norm_vec1 * norm_vec2)


class Document:
    """
    Represents a document in a specific language.
    """
    def __init__(self, content: str, lang: Lang, id: int, preprocess_func, args):
        self.tokens = preprocess_func(content, lang, args)
        self.lang = lang
        self.id = id
    
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

def batch_load_documents(executor, divide_batch, args, path=CORPUS_PATH, preprocess_func=simple_split):
    """
    Load documents from a file.
    """

    partial_create_doc = partial(create_doc, args=args)
    # Reading the JSON file
    batch_raw = []
    with open(path, "r") as f:
        raw_docs = ijson.items(f, "item")

        for raw_doc in raw_docs:
            batch_raw.append(raw_doc)
            if len(batch_raw) == LOAD_BATCH_SIZE:
                yield list(executor.map(partial_create_doc, batch_raw, chunksize=LOAD_BATCH_SIZE // divide_batch))
                batch_raw = []
            
            
        if len(batch_raw) > 0:
            yield list(executor.map(partial_create_doc, batch_raw, chunksize=LOAD_BATCH_SIZE // divide_batch))


def save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def create_doc(raw_doc, args):
    return Document(raw_doc["text"], Lang(raw_doc["lang"]), raw_doc["docid"], preprocess_text, args)


def load_lazy_arrays(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def get_args():
    parser = argparse.ArgumentParser(description="Process command-line arguments.")

    parser.add_argument('--lemmatize', type=bool, default=False, help="Whether to lematize the documents or not.")
    parser.add_argument('--stem', type=bool, default=False, help="Whether to stem the documents or not.")
    parser.add_argument('--stopwords', type=bool, default=False, help="Whether to remove stopwords from the documents or not.")
    parser.add_argument('--lowercase', type=bool, default=False, help="Whether to lowercase the documents or not.")
    parser.add_argument('--remove_punctuation', type=bool, default=False, help="Whether to remove punctuation from the documents or not.")
    parser.add_argument('--remove_numbers', type=bool, default=False, help="Whether to remove numbers from the documents or not.")
    parser.add_argument('--remove_special_characters', type=bool, default=False, help="Whether to remove special characters from the documents or not.")
    parser.add_argument('--use_prob_idf', type=bool, default=False, help="Whether to use probabilistic idf or not.")
    parser.add_argument('--use_tf_log_ave', type=bool, default=False, help="Whether to use log average tf or not.")
    parser.add_argument('--uset_tf_augmented', type=bool, default=False, help="Whether to use augmented tf or not.")
    parser.add_argument('--use_tf_boolean', type=bool, default=False, help="Whether to use boolean tf or not.")
    parser.add_argument('--use_tf_log', type=bool, default=False, help="Whether to use log tf or not.")
    parser.add_argument('--use_normalization_pivot', type=bool, default=False, help="Whether to use pivot normalization or not.")
    parser.add_argument('--idf_save_path', type=str, required=True, help="The path to save the idf scores.")
    parser.add_argument('--tf_idf_save_path', type=str, required=True, help="The path to save the tf-idf scores.")
    parser.add_argument('--vocab_path', type=str, default=None, help="The path to load the vocabulary.")
    parser.add_argument('--vocab_save_path', type=str, default='.cache/vocabulary_raw.pkl', help="The path to save the vocabulary, if it does not exist yet")
    parser.add_argument('--vocab_mapping_save_path', type=str, default='.cache/vocabulary_mapping.pkl', help="The path to save the vocabulary, if it does not exist yet")
    parser.add_argument('--tf_idf_arrays_save_path', type=str, default='.cache/arrays.pkl', help="The path to save the vocabulary, if it does not exist yet")
    parser.add_argument('--inference', type=bool, default=False, help="Whether to run inference or not.")
    parser.add_argument('--idf_path', type=str, default=None, help="The path to load the idf scores.")
    parser.add_argument('--tf_idf_path', type=str, default=None, help="The path to load the tf-idf scores.")


    # Parse the arguments
    args = parser.parse_args()
    return args
