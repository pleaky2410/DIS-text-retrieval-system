from collections import defaultdict
import pickle
import os
import numpy as np
from utils import Document, Query, cossine_similarity, compute_vocabulary
import heapq
from concurrent.futures import ProcessPoolExecutor
import utils
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def compute_tf(doc: Document):
    """
    Compute the term frequency (TF) for each word in a document.

    Parameters:
    doc (str): The input document as an object Document.

    Returns:
    dict: A dictionary where keys are words and values are their normalized term frequencies.
    """
    tf = defaultdict(float)
    words = doc.getTokens()

    # Count occurrences of each word
    for word in words:
        tf[word] += 1

   # normalize by the maximum term frequency
    max_value = max(tf.values()) 

    tf = {word: tf[word] / max_value for word in tf}

    return doc.getId(), tf

def compute_idf(documents: list, save_path="./.cache/idf.scores"):
    """
    Compute the inverse document frequency (IDF) for each unique word across all documents.

    Parameters:
    documents (list): A list of documents (Document).

    Returns:
    dict: A dictionary where keys are words and values are their IDF scores.
    """

    idf = defaultdict(float)
    total_docs = len(documents)

    # Count unique occurrences of words in each document
    for doc in documents:
        unique_words = set(doc.getTokens())

        for word in unique_words:
            idf[word] += 1

    # Calculate IDF for each word
    for word in idf:
        idf[word] = np.log(total_docs / (1 + idf[word]))  # Adding 1 to avoid division by zero
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(idf, f)

    return idf




def compute_unique_words(documents: list):
    """
    Compute the set of unique words across a batch of documents.

    Parameters:
    documents (list): A list of documents (strings).

    Returns:
    list: A list of sets containing all unique words in the documents.
    """

    unique_words = [set(doc.getTokens()) for doc in documents]
    return unique_words

def compute_tfidf(documents: list, save_path="./.cache/tf-idf.scores"):
    """
    Compute the TF-IDF scores for each document in a list of documents.

    Parameters:
    documents (list): A list of documents (strings).

    Returns:
    list: A list of dictionaries, each containing TF-IDF scores for the corresponding document.
    """

    # TODO: remove this in the end to get better performance
    assert isinstance(documents, list) and all(isinstance(doc, Document) for doc in documents), "Input must be a list of Document objects"

    tfidf = defaultdict(dict)
    idf = compute_idf(documents)

    with ProcessPoolExecutor() as executor:
        tfs = list(executor.map(compute_tf, documents))
    
    
    for tf, doc_id in tfs:
        doc_tfidf = {word: tf[word] * idf[word] for word in tf}
        tfidf[doc_id] = doc_tfidf
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(tfidf, f)

    return tfidf


def compute_tfidf_query(query: Query, idf_scores: dict):
    """
    Compute the TF-IDF scores for a query.

    Parameters:
    query (Query): The input query.

    Returns:
    dict: A dictionary containing TF-IDF scores for the query.
    """
    # TODO: remove this in the end to get better performance
    assert isinstance(query, Query), "Input must be a Query object"
    assert all(isinstance(score, float) for score in idf_scores.values()), "IDF scores must be floats"

    
    tokens = query.getTokens()
    result = defaultdict(float)

    for token in tokens:
        result[token] += 1
    
    max_value = max(result.values())
    result = {word: result[word] * idf_scores[word] / max_value for word in result}

    return result

def tf_idf_get_query_ranking(query: Query, documents: list, idf_scores: dict, tfidf_scores: dict, vocabulary: set, idf_scores_path="./.cache/idf.scores", tfidf_scores_path="./.cache/tf-idf.scores", vocabulary_path="./.cache/vocabulary"):
    """
    Compute the ranking of documents according to the TF-IDF scores of a query.

    Parameters:
    query (Query): The input query.
    documents (list): A list of documents (Document).
    idf_scores (dict): A dictionary containing IDF scores for each word.

    Returns:
    list: A list of tuples, each containing a document and its corresponding TF-IDF score.
    """

    def get_vectors(dict1, dict2):
        """
        Get vectors of the same length for two dictionaries by filling in missing keys with zeros.
        """
        vec1 = [dict1.get(key, 0) for key in vocabulary]
        vec2 = [dict2.get(key, 0) for key in vocabulary]
        return vec1, vec2

    if not idf_scores:
        try:
            with open(idf_scores_path, "rb") as f:
                idf_scores = pickle.load(f)
        except FileNotFoundError:
            print("IDF scores not found. Computing them now...")
            idf_scores = compute_idf(documents, idf_scores_path)
        
    if not tfidf_scores:
        try:
            with open(tfidf_scores_path, "rb") as f:
                tfidf_scores = pickle.load(f)
        except FileNotFoundError:
            print("TF-IDF scores not found. Computing them now...")
            tfidf_scores = compute_tfidf(documents, tfidf_scores_path)
    
    if not vocabulary:
        try:
            with open(vocabulary_path, "rb") as f:
                vocabulary = pickle.load(f)
        except FileNotFoundError:
            print("Vocabulary not found. Computing it now...")
            vocabulary = compute_vocabulary(documents, vocabulary_path)
        
    query_tfidf = compute_tfidf_query(query, idf_scores)

    ranking = []
    for doc in documents:
        doc_id = doc.getId()
        doc_tfidf = tfidf_scores[doc_id]
        vec1, vec2 = get_vectors(query_tfidf, doc_tfidf)
        similarity = cossine_similarity(vec1, vec2)
        if len(ranking) < 10:
            heapq.heappush(ranking, (similarity, doc))
        else:
            if similarity > ranking[0][0]:
                heapq.heappop(ranking)
                heapq.heapreplace(ranking, (similarity, doc.getId()))
        

    ranking = [doc for _, doc in ranking]

    return sorted(ranking, reverse=True)
        
IDF_SAVE_PATH = './.cache/idf_scores.pkl'
TF_IDF_SAVE_PATH = './.cache/tf_idf_docs.pkl'
        

if __name__ == "__main__"  :
    idf  = defaultdict(float)
    tf_docs = defaultdict(list)
    total_docs = 0
    with ProcessPoolExecutor() as executor:
        for batch in tqdm(utils.batch_load_documents(executor)):
            total_docs += len(batch)
            
            list_unique_words =  compute_unique_words(batch)

            for unique_words in list_unique_words:
                for unique_word in unique_words:
                    idf[unique_word] += 1
            
            tfs_docs_id = list(executor.map(compute_tf, batch, chunksize=utils.LOAD_BATCH_SIZE // 16))
            for doc_id, tf_doc in tfs_docs_id:
                for word, tf in tf_doc.items():
                    if word not in tf_docs:
                        tf_docs[word] = [(doc_id, tf)]
                    else:
                        tf_docs[word].append((doc_id, tf))
            del tfs_docs_id
            print(f"Processed {total_docs} documents")
            

    idf = {word: np.log(total_docs / (1 + idf[word])) for word in idf}

    utils.save(IDF_SAVE_PATH, idf)

    # tf_docs = {doc_id: {word: tf * idf[word] for word, tf in tf_doc.items()} for doc_id, tf_doc in tf_docs.items()}
    # utils.save(TF_IDF_SAVE_PATH, tf_docs)



