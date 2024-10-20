from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from functools import partial
import os
from typing import Dict, List, NamedTuple
import jax.lax
from tqdm.auto import tqdm
import numpy as np
import math
import scipy.sparse as sp

# Specify data types for memory efficiency and performance
FLOAT_TYPE = "float32"
INT_TYPE = "int32"

class TokenData(NamedTuple):
    ids: List[List[int]]
    vocab: Dict[str, int]

class SearchResults(NamedTuple):
    documents: np.ndarray
    scores: np.ndarray

def compute_doc_frequencies(tokenized_corpus, unique_tokens, show_progress=True) -> dict:
    """
    Compute document frequencies (DF) for each token in the corpus.

    Parameters:
    tokenized_corpus (List[List[int]]): List of tokenized documents.
    unique_tokens (List[int]): List of unique token IDs.
    show_progress (bool): Whether to show a progress bar.

    Returns:
    dict: Dictionary with token IDs as keys and document frequencies as values.
    """
    unique_tokens = set(unique_tokens)
    doc_freqs = {token: 0 for token in unique_tokens}
    for doc_tokens in tqdm(tokenized_corpus, disable=not show_progress, desc="Counting Tokens"):
        for token in unique_tokens.intersection(doc_tokens):
            doc_freqs[token] += 1
    return doc_freqs

def create_idf_array(doc_freqs: dict, total_docs: int) -> np.ndarray:
    """
    Compute the Inverse Document Frequency (IDF) for each token using the document frequencies.

    Parameters:
    doc_freqs (dict): Dictionary with token IDs as keys and document frequencies as values.
    total_docs (int): Total number of documents in the corpus.

    Returns:
    np.ndarray: Array of IDF values.
    """
    idf_array = np.zeros(len(doc_freqs), dtype=FLOAT_TYPE)
    for token_id, df in doc_freqs.items():
        idf_array[token_id] = math.log(1 + (total_docs - df + 0.5) / (df + 0.5)) # Lucene variant
    return idf_array

def compute_term_frequency(tf_array, doc_len, avg_doc_len, k1, b):
    """
    Compute term frequency using the BM25 formula.

    Parameters:
    tf_array (np.ndarray): Array of term frequencies.
    doc_len (int): Length of the document.
    avg_doc_len (float): Average document length in the corpus.
    k1 (float): BM25 parameter.
    b (float): BM25 parameter.

    Returns:
    np.ndarray: Array of term frequency values.
    """
    return tf_array / (k1 * ((1 - b) + b * doc_len / avg_doc_len) + tf_array) # Robertson variant

def get_token_counts(token_ids):
    """
    Get token counts from a list of token IDs.

    Parameters:
    token_ids (List[int]): List of token IDs.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Arrays of unique token IDs and their counts.
    """
    token_counter = Counter(token_ids)
    return np.array(list(token_counter.keys()), dtype=INT_TYPE), np.array(list(token_counter.values()), dtype=FLOAT_TYPE)

def create_score_matrix(corpus_token_ids, idf_array, avg_doc_len, doc_freqs, k1, b, show_progress=True):
    """
    Create the BM25 score matrix for the corpus.
    Compute the BM25 scores for each token in each document of the corpus, the scores along with 
    the corresponding document and vocabulary indices.

    Parameters:
    corpus_token_ids (List[List[int]]): List of tokenized documents.
    idf_array (np.ndarray): Array of IDF values.
    avg_doc_len (float): Average document length in the corpus.
    doc_freqs (dict): Dictionary with token IDs as keys and document frequencies as values.
    k1 (float): BM25 parameter.
    b (float): BM25 parameter.
    show_progress (bool): Whether to show a progress bar.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of scores, document indices, and vocabulary indices.
    """
    array_size = sum(doc_freqs.values())
    
    # Initialize arrays to store scores, document indices, and vocabulary indices
    scores = np.empty(array_size, dtype=FLOAT_TYPE)
    doc_indices = np.empty(array_size, dtype=INT_TYPE)
    vocab_indices = np.empty(array_size, dtype=INT_TYPE)
    
    i = 0
    # Iterate over each document in the corpus
    for doc_idx, token_ids in enumerate(tqdm(corpus_token_ids, desc="Computing Scores", disable=not show_progress)):
        doc_len = len(token_ids)
        
        # Get term frequencies in the current document
        vocab_indices_doc, tf_array = get_token_counts(token_ids)
        
        # Compute BM25 scores (for each token) for the current document
        scores_doc = idf_array[vocab_indices_doc] * compute_term_frequency(tf_array, doc_len, avg_doc_len, k1, b)
        
        doc_len = len(scores_doc)
        
        # Determine the start and end indices for the current document scores in the arrays
        start = i
        end = i + doc_len
        i = end  # position where next document's scores will start
        
        # Store the computed scores and corresponding indices in the arrays
        doc_indices[start:end] = doc_idx
        vocab_indices[start:end] = vocab_indices_doc
        scores[start:end] = scores_doc
    
    return scores, doc_indices, vocab_indices

def tokens_to_strings(token_data: TokenData) -> List[List[str]]:
    """
    Convert token IDs back to strings.

    Parameters:
    token_data (TokenData): TokenData object containing token IDs and vocabulary.

    Returns:
    List[List[str]]: List of documents with token strings.
    """
    reverse_vocab = {v: k for k, v in token_data.vocab.items()}
    return [[reverse_vocab[token_id] for token_id in doc_ids] for doc_ids in token_data.ids]

def get_top_k(query_scores, k):
    """
    Get the top k scores and their indices.

    Parameters:
    query_scores (np.ndarray): Array of query scores.
    k (int): Number of top scores to retrieve.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Arrays of top k scores and their indices.
    """
    topk_scores, topk_indices = jax.lax.top_k(query_scores, k)
    return np.asarray(topk_scores), np.asarray(topk_indices)

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        """
        Initialize the BM25 object with parameters.

        Parameters:
        k1 (float): BM25 parameter.
        b (float): BM25 parameter.
        """
        self.k1 = k1
        self.b = b

    def compute_relevance_scores(self, query_token_ids):
        """
        Compute relevance scores for a query. Using precomputed BM25 scores.

        Parameters:
        data (np.ndarray): Array of score data.
        indptr (np.ndarray): Index pointer array.
        indices (np.ndarray): Array of indices.
        num_docs (int): Number of documents.
        query_token_ids (np.ndarray): Array of query token IDs.

        Returns:
        np.ndarray: Array of relevance scores for the query
        """
        data = self.scores["data"] 
        indptr = self.scores["indptr"]
        indices = self.scores["indices"]
        num_docs = self.scores["num_docs"]
        scores = np.zeros(num_docs, dtype=FLOAT_TYPE)
        for i in range(len(query_token_ids)):
            start, end = indptr[query_token_ids[i]], indptr[query_token_ids[i] + 1]
            np.add.at(scores, indices[start:end], data[start:end])
        return scores

    def build_index(self, unique_token_ids, corpus_token_ids, show_progress=True):
        """
        Build the index for the corpus.

        Parameters:
        unique_token_ids (List[int]): List of unique token IDs.
        corpus_token_ids (List[List[int]]): List of tokenized documents.
        show_progress (bool): Whether to show a progress bar.

        Returns:
        dict: Dictionary containing score matrix data, indices, index pointer, and number of documents.
        """
        avg_doc_len = np.mean([len(doc_ids) for doc_ids in corpus_token_ids])
        total_docs = len(corpus_token_ids)
        doc_freqs = compute_doc_frequencies(corpus_token_ids, unique_token_ids, show_progress)
        idf_array = create_idf_array(doc_freqs, total_docs)
        scores, doc_indices, vocab_indices = create_score_matrix(corpus_token_ids, idf_array, avg_doc_len, doc_freqs, self.k1, self.b, show_progress)
        score_matrix = sp.csc_matrix((scores, (doc_indices, vocab_indices)), shape=(total_docs, len(unique_token_ids)), dtype=FLOAT_TYPE) # efficiently stores BM25 scores for each term in each document
        return {"data": score_matrix.data, "indices": score_matrix.indices, "indptr": score_matrix.indptr, "num_docs": total_docs}

    def index_corpus(self, corpus: TokenData, show_progress=True):
        """
        Index the corpus.

        Parameters:
        corpus (TokenData): TokenData object containing tokenized documents and vocabulary.
        show_progress (bool): Whether to show a progress bar.
        """
        self.scores = self.build_index(list(corpus.vocab.values()), corpus.ids, show_progress)
        self.vocab_dict = corpus.vocab

    def get_token_ids(self, query_tokens: List[str]) -> List[int]:
        """
        Get token IDs for a query.

        Parameters:
        query_tokens (List[str]): List of query tokens.

        Returns:
        List[int]: List of token IDs.
        """
        return [self.vocab_dict[token] for token in query_tokens if token in self.vocab_dict]

    def retrieve_top_k(self, query_tokens: List[str], k: int = 10):
        """
        Retrieve the top k documents for a query.

        Parameters:
        query_tokens (List[str]): List of query tokens.
        k (int): Number of top documents to retrieve.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of top k scores and their indices.
        """
        query_tokens_ids = self.get_token_ids(query_tokens)
        scores = self.compute_relevance_scores(np.asarray(query_tokens_ids, dtype=INT_TYPE))
        return get_top_k(scores, k)

    def search(self, queries: TokenData, k: int = 10, show_progress: bool = True, n_threads: int = 0, chunksize: int = 50):
        """
        Retrieve the top-k documents for each query.

        Parameters:
        queries (TokenData): A TokenData object with tokenized queries.
        k (int): Number of documents to retrieve for each query.
        show_progress (bool): Whether to show a progress bar.
        n_threads (int): Number of jobs to run in parallel. If -1, it will use all available CPUs. If 0, it will run the jobs sequentially.
        chunksize (int): Number of batches to process in each job in the multiprocessing pool.

        Returns:
        Tuple of top k document ids retrieved and their scores.
        """
        if n_threads == -1:
            n_threads = os.cpu_count()
        queries = tokens_to_strings(queries)
        topk_fn = partial(self.retrieve_top_k, k=k)
        if n_threads == 0:
            out = list(tqdm(map(topk_fn, queries), total=len(queries), desc="Retrieving Documents", disable=not show_progress))
        else:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                out = list(tqdm(executor.map(topk_fn, queries, chunksize=chunksize), total=len(queries), desc="Retrieving Documents", disable=not show_progress))
        scores, indices = zip(*out)
        return SearchResults(documents=np.array(indices), scores=np.array(scores))