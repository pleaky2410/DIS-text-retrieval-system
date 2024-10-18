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
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# assumes all queries are from the same lang
# TODO: this function takes an awful amount of time for german for some reason
def vectorize_query_batch(query_batch, idf):
    manager = tf_idf_training.COOMatrixManager(query_batch[0].getLang())
    empty_query_indices = []

    # query_batch is guaranteed to have at most BATCH_SIZE queries
    with ProcessPoolExecutor() as executor:
        if args.use_tf_log_ave:
            tfs_queries_id = list(executor.map(tf_idf_training.compute_log_average_tf, query_batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))
        elif args.use_tf_augmented:
            tfs_queries_id = list(executor.map(tf_idf_training.compute_augmented_tf, query_batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))
        elif args.use_tf_boolean:
            tfs_queries_id = list(executor.map(tf_idf_training.compute_boolean_tf, query_batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))
        elif args.use_tf_log:
            tfs_queries_id = list(executor.map(tf_idf_training.compute_logarithm_tf, query_batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))
        else:
            tfs_queries_id = list(executor.map(tf_idf_training.compute__basic_tf, query_batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))
        
        for query_id, _, tf_query in tfs_queries_id:
            if len(tf_query) == 0:
                empty_query_indices.append(query_id)
                continue
            for int_word, tf in tf_query.items():
                manager.add(query_id, int_word, tf * idf[int_word])
        # tokens = query.getTokens()
        # if len(tokens) == 0:
        #     print("Empty query: ", query.getId())
        #     empty_query_indices.append(query.getId())
        #     continue
        # freqs = Counter(tokens)
        # max_freq = max(freqs.values())

        # for word, freq in tqdm(freqs.items(), mininterval=3, desc="Words", leave=False):
        #     int_word = tf_idf_training.get_int(word, query.getLang())
        #     manager.add(query.getId(), int_word, freq / max_freq * idf[int_word])
    
    return manager.get_csr_matrix(), manager.get_docid_row_idx_mapping(), empty_query_indices

    
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

            # get tf-idf norms for every document (rows)
            tf_idf_norms = np.sqrt(tf_idf.power(2).sum(axis=1)).A1  # Norms of all rows
            len_query_list = len(query_list)
            for i in range(0, len_query_list, utils.LOAD_BATCH_SIZE):
                max_idx = min(i + utils.LOAD_BATCH_SIZE, len_query_list)

                # vectorize batch of queries and get the norm for every query (row)
                query_batch_matrix, query_id_mapping, empty_query_indices = vectorize_query_batch(query_list[i:max_idx], idf)
                query_batch_matrix_norms = np.sqrt(query_batch_matrix.power(2).sum(axis=1)).A1

                # calculate cosine similarities
                dot_products = tf_idf.dot(query_batch_matrix.T).toarray()  
                det = np.outer(tf_idf_norms, query_batch_matrix_norms)
                det[det == 0] = 1e-10
                cosine_similarities = dot_products / det 

                # choose the best 10 for each query
                top10_indices = np.argpartition(cosine_similarities, -10, axis=0)[-10:]
                for i in range(top10_indices.shape[1]):
                    writer.writerow([query_id_mapping[i], [docid_row_mapping[j] for j in top10_indices[:, i]]])
                
                for i in empty_query_indices:
                    writer.writerow([i, random.sample(list(docid_row_mapping.values()), 10)])

            del tf_idf
            del docid_row_mapping
            del idf
            del tf_idf_norms
            gc.collect()
    

    end = time.time()
    print("Time taken: ", end - start)





        
     

        

