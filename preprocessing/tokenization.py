import re
from typing import Dict, List, Union, NamedTuple, Tuple
from tqdm.auto import tqdm
import Stemmer

from .stopwords import (
    STOPWORDS_EN,
    STOPWORDS_EN_PLUS,
    STOPWORDS_GERMAN,
    STOPWORDS_DUTCH,
    STOPWORDS_FRENCH,
    STOPWORDS_SPANISH,
    STOPWORDS_PORTUGUESE,
    STOPWORDS_ITALIAN,
    STOPWORDS_RUSSIAN,
    STOPWORDS_SWEDISH,
    STOPWORDS_NORWEGIAN,
    STOPWORDS_CHINESE,
    STOPWORDS_ARABIC,
    STOPWORDS_KOREAN
)

from konlpy.tag import Kkma

class KoreanStemmer:
    def __init__(self):
        self.kkma = Kkma()

    def stem_words(self, words):
        return [self.kkma.morphs(word) for word in words]

stemmers = {
    "en": Stemmer.Stemmer("english"),
    "fr": Stemmer.Stemmer("french"),
    # "es": Stemmer.Stemmer("spanish"),
    "de": Stemmer.Stemmer("german"),
    "it": Stemmer.Stemmer("italian"),
    "ar": Stemmer.Stemmer("arabic"),
    "ko": KoreanStemmer()
}

class TokenData(NamedTuple):
    ids: List[List[int]]
    vocab: Dict[str, int]

def get_stopwords(language: str) -> List[str]:
    """
    Get stopwords for a given language.
    """
    if language in ["english", "en"]: 
        return STOPWORDS_EN
    elif language in ["english_plus", "en_plus"]:
        return STOPWORDS_EN_PLUS
    elif language in ["german", "de"]:
        return STOPWORDS_GERMAN
    elif language in ["dutch", "nl"]:
        return STOPWORDS_DUTCH
    elif language in ["french", "fr"]:
        return STOPWORDS_FRENCH
    # elif language in ["spanish", "es"]:
    #     return STOPWORDS_SPANISH
    elif language in ["portuguese", "pt"]:
        return STOPWORDS_PORTUGUESE
    elif language in ["italian", "it"]:
        return STOPWORDS_ITALIAN
    elif language in ["russian", "ru"]:
        return STOPWORDS_RUSSIAN
    elif language in ["swedish", "sv"]:
        return STOPWORDS_SWEDISH
    elif language in ["norwegian", "no"]:
        return STOPWORDS_NORWEGIAN
    elif language in ["chinese", "zh"]:
        return STOPWORDS_CHINESE
    elif language in ["arabic", "ar"]:
        return STOPWORDS_ARABIC
    elif language in ["korean", "ko"]:
        return STOPWORDS_KOREAN
    else:
        # print(f"{language} stopwords not supported, defaulting to English stopwords")
        return STOPWORDS_EN

def tokenize(
    texts: List[Tuple[str, str]],  # list of tuples (text, language)
    lower: bool = True,
    return_ids: bool = True,
    show_progress: bool = True,
    leave: bool = False,
) -> Union[List[List[str]], TokenData]:
    """
    Tokenize a list of texts with optional stemming and stopwords removal.

    Parameters
    ----------
    texts : List[Tuple[str, str]]
        A list of tuples where each tuple contains a text and its language.

    lower : bool, optional
        Convert text to lowercase before tokenization.

    return_ids : bool, optional
        Return token IDs and vocabulary if True, else return tokenized strings.

    show_progress : bool, optional
        Show progress bar if True.

    leave : bool, optional
        Leave progress bar after completion if True.

    Returns
    -------
    Union[List[List[str]], TokenData]
        Tokenized texts as strings or token IDs with vocabulary.
    """
    token_pattern = r"(?u)\b\w\w+\b"
    split_fn = re.compile(token_pattern).findall

    corpus_ids = []
    token_to_index = {}

    for text, language in tqdm(texts, desc="Tokenizing texts", leave=leave, disable=not show_progress):
        stopwords_set = set(get_stopwords(language))
        if lower:
            text = text.lower()

        tokens = split_fn(text)
        doc_ids = []

        for token in tokens:
            if token in stopwords_set:
                continue

            if token not in token_to_index:
                token_to_index[token] = len(token_to_index)

            token_id = token_to_index[token]
            doc_ids.append(token_id)

        corpus_ids.append(doc_ids)

    unique_tokens = list(token_to_index.keys())

    stemmer = stemmers.get(language, Stemmer.Stemmer("english"))
    if hasattr(stemmer, "stemWords"):
        stemmer_fn = stemmer.stemWords
    elif callable(stemmer):
        stemmer_fn = stemmer
    else:
        raise ValueError("Stemmer must have a `stemWords` method or be callable")

    stemmed_tokens = stemmer_fn(unique_tokens)
    vocab = set(stemmed_tokens)
    vocab_dict = {token: i for i, token in enumerate(vocab)}
    stem_id_to_stem = {v: k for k, v in vocab_dict.items()}
    doc_id_to_stem_id = {token_to_index[token]: vocab_dict[stem] for token, stem in zip(unique_tokens, stemmed_tokens)}

    for i, doc_ids in enumerate(tqdm(corpus_ids, desc="Stemming tokens", leave=leave, disable=not show_progress)):
        corpus_ids[i] = [doc_id_to_stem_id[doc_id] for doc_id in doc_ids]

    if return_ids:
        return TokenData(ids=corpus_ids, vocab=vocab_dict)
    else:
        reverse_dict = stem_id_to_stem if stemmers is not None else unique_tokens
        for i, token_ids in enumerate(tqdm(corpus_ids, desc="Reconstructing token strings", leave=leave, disable=not show_progress)):
            corpus_ids[i] = [reverse_dict[token_id] for token_id in token_ids]
        return corpus_ids