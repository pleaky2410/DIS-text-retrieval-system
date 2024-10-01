import utils
import numpy as np
import os
import pickle

INFERENCE_BATCH_SIZE = 20000

def get_doc_ids(tf_idf):
    doc_ids = set()
    for word in tf_idf:
        doc_ids.update(tf_idf[word].keys())
    
    return list(doc_ids)


def get_vector(tf_idf, doc_id):
    """
    Get vectors of the same length for two dictionaries by filling in missing keys with zeros.
    """
    # Create vec1: The document vector based on tf-idf scores
    vec1 = np.array([tf_idf[word][doc_id] if word in tf_idf and doc_id in tf_idf[word] else 0 for word in vocabulary])

    # Add the 0 for vec1 as in the original code
    vec1 = np.append(vec1, 0)


if __name__=="__main__":
    args = utils.get_args()
    vocabulary = utils.load(args.vocab_path)

    if os.path.exists(args.tf_idf_arrays_save_path):
        raise ValueError("The save path already exists. Please remove it before running the script.")
    tf_idf = utils.load(args.tf_idf_path)
    doc_ids = get_doc_ids(tf_idf)
    len_doc_ids = len(doc_ids)

    for i in range(0, len_doc_ids, INFERENCE_BATCH_SIZE):
        last_idx = min(i + INFERENCE_BATCH_SIZE, len_doc_ids)
        size = last_idx - i
        batch_arrays = []
        for j in range(i, min(i + INFERENCE_BATCH_SIZE, len_doc_ids)):
            doc_id = doc_ids[j]
            vec1 = get_vector(tf_idf, doc_id)
            batch_arrays.append(vec1)
        
        result = np.vstack(batch_arrays)


        os.makedirs(os.path.dirname(args.tf_idf_arrays_save_path), exist_ok=True)
        with open(args.tf_idf_arrays_save_path, "ab") as f:
            pickle.dump(result, f)


            
    
    
        



