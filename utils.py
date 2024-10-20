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
import string


CORPUS_PATH = "data/corpus.json"
DEV_PATH = "data/dev.csv"
TEST_PATH = "data/test.csv"
TRAIN_PATH = "data/train.csv"

class DocProcessingConfig():
    __slots__ = ['lemmatize', 'stopwords', 'lowercase', 'remove_punctuation', 'remove_numbers', 'remove_special_chars']

    def __init__(self,
                    lemmatize: bool = False,
                    stopwords: bool = False,
                    lowercase: bool = False,
                    remove_punctuation: bool = False,
                    remove_numbers: bool = False,
                    remove_special_chars: bool = False,
                    **kwargs
                 ) -> None:
        
        self.lemmatize = lemmatize
        self.stopwords = stopwords
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_special_chars = remove_special_chars

        for key, value in kwargs.items():
            setattr(self, key, value)


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
    GERMAN = 'de'
    FRENCH = 'fr'
    SPANISH = 'es'
    ITALIAN = 'it'
    ARABIC = 'ar'
    ENGLISH = 'en'
    KOREAN = 'ko'


def args_to_doc_processing_config(args):
    return DocProcessingConfig(
        lemmatize=args.lemmatize,
        stopwords=args.stopwords,
        lowercase=args.lowercase,
        remove_punctuation=args.remove_punctuation,
        remove_numbers=args.remove_numbers,
        remove_special_chars=args.remove_special_chars
    )


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

def is_korean_char(char):
    return ('\uAC00' <= char <= '\uD7A3') or ('\u1100' <= char <= '\u11FF') or ('\u3130' <= char <= '\u318F')

def is_arabic_char(char):
    return (
        ('\u0600' <= char <= '\u06FF') or  # Arabic main block
        ('\u0750' <= char <= '\u077F') or  # Arabic Supplement
        ('\u08A0' <= char <= '\u08FF') or  # Arabic Extended-A
        ('\uFB50' <= char <= '\uFDFF') or  # Arabic Presentation Forms-A
        ('\uFE70' <= char <= '\uFEFF')     # Arabic Presentation Forms-B
    )

def preprocess_text(text: str, lang: Lang, config: DocProcessingConfig):
    """
    Preprocess the input text by lowercasing, removing punctuation, 
    tokenizing, removing stop words, and stemming.

    Parameters:
    text (str): The input text to preprocess.

    Returns:
    list: A list of processed tokens.
    """
    # Lowercase the text and remove punctuation in a single step
    if config.lowercase:
        text = text.lower()
    
    if config.remove_numbers:
        number_pattern = r'\d+'  # Matches any sequence of digits
        date_pattern = r'(\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}\s\w+\s\d{4}\b|\b\w+\s\d{1,2},\s\d{4}\b)'
        text = re.sub(number_pattern, ' ', text)
        text = re.sub(date_pattern, ' ', text)
    
    if config.remove_special_chars:
        pattern = r'[#\$%\&\/\(\)={}\[\]\\\.\|\n\t\-]'
        text = re.sub(pattern, ' ', text).strip()

    
    if lang == Lang.ARABIC:
        punctuation_pattern = r'[،؛؟.!?…“”"\'(){}[\]«»]'
        if config.remove_punctuation:
            text = re.sub(punctuation_pattern, ' ', text).strip()
        tokens = nltk.word_tokenize(text)
        if config.stopwords:
            stopwords = get_stopwords(Lang.ARABIC)
            tokens = [
                token
                for token in tokens
                if token not in stopwords
            ]
        if config.lemmatize:
            stemmer = ISRIStemmer()
            for i, token in enumerate(tokens):
                tokens[i] = stemmer.stem(token)
    else :
        nlp = get_nlp_pipeline(lang)
        doc = nlp(text)
        tokens = [token for token in doc]
        if config.remove_punctuation:
            tokens = [token for token in tokens if not token.is_punct]
        
        if config.stopwords:
            tokens = [token for token in tokens if not token.is_stop]
        
        if config.lemmatize:
            tokens = [token.lemma_ for token in tokens]
        else:
            tokens = [token.text for token in tokens]
        
    for i, token in enumerate(tokens):
        tokens[i] = token.strip()
    if lang == Lang.ARABIC:
        tokens = [token for token in tokens if len(token) > 2]
    elif lang == Lang.KOREAN:
        tokens = [token for token in tokens if len(token) > 2]
    
    elif lang == Lang.ENGLISH:
        tokens = [token for token in tokens if len(token) > 2 and bool(re.fullmatch(r'[a-z0-9]+', token))]

    else:
        tokens = [token for token in tokens if len(token) > 2]

    if len(tokens) == 0:
        print(f"Empty document ({lang.value}): {text}")
        # raise ValueError("Empty document")
    return tokens

def preprocess_text_2(text:str, lang: Lang, cofing=None):
    text = text.strip()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token.lower()) for token in tokens if token not in get_stopwords(lang)]
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
    def __init__(self, content: str, lang: Lang, id: int, preprocess_func, config: DocProcessingConfig=DocProcessingConfig()):
        self.tokens = preprocess_func(content, lang, config)
        self.lang = lang
        self.id = id
        self.config = config
    
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

    def __init__(self, content: str, lang: Lang, id: int, preprocess_func, config: DocProcessingConfig=DocProcessingConfig()):
        self.tokens = preprocess_func(content,lang, config)
        self.lang = lang
        self.content = content
        self.lang = lang
        self.id = id
    
    def getLang(self):
        return self.lang
    
    def getTokens(self):
        return self.tokens
    
    def getId(self):
        return self.id
    
    
def simple_split(content):
    return content.split()

def batch_load_documents(executor, divide_batch, config: DocProcessingConfig=DocProcessingConfig(), path=CORPUS_PATH, lang: None | Lang=None):
    """
    Load documents from a file.
    """

    partial_create_doc = partial(create_doc, config=config)
    # Reading the JSON file
    batch_raw = []
    # files = 0
    with open(path, "r") as f:
        raw_docs = ijson.items(f, "item")

        for raw_doc in raw_docs:
            # if raw_doc["lang"] != "de":
            #     continue
            if lang and lang.value != raw_doc["lang"]:
                continue
            # files += 1
                
            batch_raw.append(raw_doc)

            if len(batch_raw) == LOAD_BATCH_SIZE:
                if executor is not None:
                    yield list(executor.map(partial_create_doc, batch_raw, chunksize=LOAD_BATCH_SIZE // divide_batch))
                else:
                    yield [partial_create_doc(raw_doc=d) for d in batch_raw]

                batch_raw = []
                gc.collect()

            # if files == 1000:
            #     yield list(executor.map(partial_create_doc, batch_raw, chunksize=LOAD_BATCH_SIZE // divide_batch))
            #     batch_raw = []
            #     break
            
            
        if len(batch_raw) > 0:
            if executor is not None:
                yield list(executor.map(partial_create_doc, batch_raw, chunksize=LOAD_BATCH_SIZE // divide_batch))
            else:
                yield [partial_create_doc(raw_doc=d) for d in batch_raw]

def save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def create_doc(raw_doc, config):
    return Document(raw_doc["text"], Lang(raw_doc["lang"]), raw_doc["docid"], preprocess_text, config)

def create_query(raw_query, config):
    return Query(raw_query["query"], Lang(raw_query["lang"]), raw_query["id"], preprocess_text, config)


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
    parser.add_argument('--remove_special_chars', type=bool, default=False, help="Whether to remove special characters from the documents or not.")
    parser.add_argument('--use_prob_idf', type=bool, default=False, help="Whether to use probabilistic idf or not.")
    parser.add_argument('--use_tf_log_ave', type=bool, default=False, help="Whether to use log average tf or not.")
    parser.add_argument('--use_tf_augmented', type=bool, default=False, help="Whether to use augmented tf or not.")
    parser.add_argument('--use_tf_boolean', type=bool, default=False, help="Whether to use boolean tf or not.")
    parser.add_argument('--use_tf_log', type=bool, default=False, help="Whether to use log tf or not.")
    parser.add_argument('--use_normalization_pivot', type=bool, default=False, help="Whether to use pivot normalization or not.")
    parser.add_argument('--idf_save_path', type=str, required=True, help="The path to save the idf scores.")
    parser.add_argument('--tf_idf_save_path', type=str, required=True, help="The path to save the tf-idf scores.")
    parser.add_argument('--vocab_save_path', type=str, default='.cache/vocabulary_raw.pkl', help="The path to save the vocabulary, if it does not exist yet")
    parser.add_argument('--vocab_mapping_save_path', type=str, default='.cache/vocabulary_mapping.pkl', help="The path to save the vocabulary, if it does not exist yet")
    parser.add_argument('--docid_row_mapping_save_path', type=str, default='.cache/doc_id_row_mapping.pkl', help="The path to save the docid_row_mapping, if it does not exist yet")
    parser.add_argument('--inference_output_save_path', type=str, default='output.csv', help="The path to save the inference output.")


    # Parse the arguments
    args = parser.parse_args()
    return args
