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
import gc

def compute__basic_tf(doc: Document)-> tuple[int, dict]:
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


def compute_boolean_tf(doc: Document):
    _, tf = compute__basic_tf(doc)
    return doc.getId(), {word: 1 if tf[word] > 0 else 0 for word in tf}


def compute_logarithm_tf(doc: Document):
    _, tf = compute__basic_tf(doc)
    return doc.getId(), {word: 1 + np.log(tf[word]) for word in tf}


def compute_augmented_tf(doc: Document):
    _, tf = compute__basic_tf(doc)
    max_tf = max(tf.values())
    return doc.getId(), {word: 0.5 + 0.5 * tf[word] / max_tf for word in tf}


def compute_log_average_tf(doc: Document):
    _, tf = compute_logarithm_tf(doc)
    avg_tf = sum(tf.values()) / len(tf)
    return doc.getId(), { word: tf[word] / (1 + np.log(avg_tf)) for word in tf}


def compute_normalization_pivot_tf(el : float, avg_number_distinct_words_in_doc: float, number_distinct_terms_in_doc: int):
    s = 0.2
    return el / ((1-s)*avg_number_distinct_words_in_doc + s*number_distinct_terms_in_doc)


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


mapping = {}
unknown_idx = 0
def get_str_int_mapping(vocabulary):
    global mapping, unknown_idx
    for i, word in enumerate(vocabulary):
        mapping[word] = i
    unknown_idx = len(vocabulary)

def get_int(word):
    return mapping[word]
    
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

def tf_idf_get_query_ranking(query: Query, idf: dict, tfidf: dict, vocabulary: set):
    """
    Compute the ranking of documents according to the TF-IDF scores of a query.

    Parameters:
    query (Query): The input query.
    documents (list): A list of documents (Document).
    idf_scores (dict): A dictionary containing IDF scores for each word.

    Returns:
    list: A list of tuples, each containing a document and its corresponding TF-IDF score.
    """

    def get_vectors(tf_idf, tf_idf_query, doc_id, vocabulary):
        """
        Get vectors of the same length for two dictionaries by filling in missing keys with zeros.
        """
        # Create vec1: The document vector based on tf-idf scores
        vec1 = np.array([tf_idf[word][doc_id] if word in tf_idf and doc_id in tf_idf[word] else 0 for word in vocabulary])
        
        # Add the 0 for vec1 as in the original code
        vec1 = np.append(vec1, 0)

        # Create vec2: The query vector based on tf-idf scores for the query
        vec2 = np.array([tf_idf_query[word] if word in tf_idf_query else 0 for word in vocabulary])
        
        # Handle unknown tokens
        unknown_tokens = set(tf_idf_query.keys()) - set(vocabulary)
        
        if unknown_tokens:
            vec2 = np.append(vec2, sum([tf_idf_query[word] for word in unknown_tokens]) / len(unknown_tokens))
        else:
            vec2 = np.append(vec2, 0)

        return vec1, vec2

        
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
    args = utils.get_args()
    ARGS = args

    num_cores = os.cpu_count() 
    divide_batch = num_cores if num_cores is not None else 16

    if args.inference:
        vocabulary = utils.load(args.vocab_path)
        idf = utils.load(args.idf_path)
        tf_idf = utils.load(args.tf_idf_path)

    if args.vocab_path is not None:
            mapping = utils.load(args.vocab_path)
            # get_str_int_mapping(vocabulary)
    else:
        vocabulary = set()
        with ProcessPoolExecutor() as executor:
            for batch in tqdm(utils.batch_load_documents(executor, divide_batch, args)):
                vocabulary.update(compute_vocabulary(batch))
        
        get_str_int_mapping(vocabulary)
        del vocabulary
        gc.collect()
        utils.save(args.vocab_save_path, mapping)
    

    idf  = defaultdict(float)
    tf_idf = {} 
    total_docs = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for batch in tqdm(utils.batch_load_documents(executor, divide_batch, args)):
            total_docs += len(batch)
            
            list_unique_words =  compute_unique_words(batch)
            number_distinct_words_in_doc = 0
            if args.use_normalization_pivot:
                number_distinct_words_in_doc += sum([len(unique_words) for unique_words in list_unique_words])

            for unique_words in list_unique_words:
                for unique_word in unique_words:
                    idf[get_int(unique_word)] += 1
            
            if args.use_tf_log_ave:
                tfs_docs_id = list(executor.map(compute_log_average_tf, batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))
            elif args.uset_tf_augmented:
                tfs_docs_id = list(executor.map(compute_augmented_tf, batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))
            elif args.use_tf_boolean:
                tfs_docs_id = list(executor.map(compute_boolean_tf, batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))
            elif args.use_tf_log:
                tfs_docs_id = list(executor.map(compute_logarithm_tf, batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))
            else:
                tfs_docs_id = list(executor.map(compute__basic_tf, batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))

            for doc_id, tf_doc in tfs_docs_id:
                for word, tf in tf_doc.items():
                    int_word = get_int(word)
                    if word not in tf_idf:
                        tf_idf[int_word] = {}
                    tf_idf[int_word][doc_id] = tf
            del tfs_docs_id
            print(f"Processed {total_docs} documents", end="\r")
            


    print("Calculating idf and tf-idf scores...")
    for word, count in tqdm(idf.items(), mininterval=5):
        int_word = get_int(word)
        if args.use_prob_idf:
            idf[int_word] = max(0, np.log((total_docs - count) / count))
        else:
            idf[int_word] = np.log(total_docs / (1 + count))
        for doc_id, tf in tf_idf[get_int(word)].items():
            tf_idf[int_word][doc_id] = tf * idf[int_word]

    del mapping
    del total_docs
    del num_cores
    del divide_batch
    gc.collect()

    print("Saving idf scores...")
    utils.save(args.idf_save_path, idf)

    save_path = args.tf_idf_save_path
    del args
    del idf
    gc.collect()

    print("Saving tf-idf scores...")
    utils.save(save_path, tf_idf)



