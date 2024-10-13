import utils
import json
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from scipy.sparse import load_npz, csr_matrix
import tf_idf_training
import numpy as np
import heapq
import pandas as pd
import gc
import csv
import time
from tqdm import tqdm

def vectorize_query(query, idf):
    manager = tf_idf_training.COOMatrixManager(query.getLang())
    freqs = Counter(query.getTokens())
    if len(freqs) == 0:
        return csr_matrix((1,idf.shape[0]), dtype=np.float32)
    max_freq = max(freqs.values())

    for word, freq in freqs.items():
        int_word = tf_idf_training.get_int(word, query.getLang())
        manager.add(0, int_word, freq / max_freq * idf[int_word])
    
    return manager.get_csr_matrix()


    
if __name__ == "__main__":
    start = time.time()
    args = utils.get_args()
    tf_idf_training.ARGS = args
    tf_idf_training.args = args
    queries = defaultdict(list)
    config = utils.args_to_doc_processing_config(args)
    partial_create_query = partial(utils.create_query, config=config)
    num_cores = os.cpu_count() 
    divide_batch = num_cores if num_cores is not None else 16
    batch_raw = []
    with open(utils.TEST_PATH, "r") as f:
        with ProcessPoolExecutor() as executor:
            df = pd.read_csv(f).to_dict(orient="records")
            for query in df:
                batch_raw.append(query)
                
                if len(batch_raw) == utils.LOAD_BATCH_SIZE:
                    batch_processed = list(executor.map(partial_create_query, batch_raw, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))

                    for query in batch_processed:
                        if len(query.getTokens()) == 0:
                            print("Empty query: ", query.getId())
                        queries[query.getLang().value].append(query)

                    batch_raw = []
    
            if len(batch_raw) > 0:
                batch_processed = list(executor.map(partial_create_query, batch_raw, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))

                for query in batch_processed:
                    queries[query.getLang().value].append(query)
    
    utils.cleanup_all()

    with open(args.inference_output_save_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "docids"])
        for lang, query_list in tqdm(queries.items(), desc="Languages"):
            print("\nLanguage: ", lang)
            print("Number of Queries: ", len(query_list))
            print()

            tf_idf =  load_npz(f"{args.tf_idf_save_path}_{lang}.npz")
            docid_row_mapping = utils.load(f"{args.docid_row_mapping_save_path}_{lang}.pkl")
            idf = np.load(args.idf_save_path + f"_{lang}.npy")
            tf_idf_norms = np.sqrt(tf_idf.multiply(tf_idf).sum(axis=1)).A1  # Norms of all rows

            for query in tqdm(query_list, desc=f"Queries for language {lang}", leave=False):
                query_vec = vectorize_query(query, idf)
                dot_products = tf_idf.dot(query_vec.T).toarray().flatten() 
                query_vec_norm = np.linalg.norm(query_vec.toarray())  # Norm of the 1-row vector
                cosine_similarities = dot_products / (tf_idf_norms * query_vec_norm)

                min_heap = []
                for i, similarity in enumerate(cosine_similarities):
                    # Push to heap if we have less than top_n elements
                    if len(min_heap) < 10:
                        heapq.heappush(min_heap, (similarity, i))  # Push a tuple of (similarity, index)
                    elif similarity > min_heap[0][0]:
                        heapq.heapreplace(min_heap, (similarity, i))  # Replace the smallest element

                writer.writerow([query.getId(), [docid_row_mapping[i] for _, i in min_heap]])
            
            del tf_idf
            del docid_row_mapping
            del idf
            del tf_idf_norms
            gc.collect()
    

    end = time.time()
    print("Time taken: ", end - start)





        
     

        

