import utils
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from scipy.sparse import load_npz
import numpy as np
import pandas as pd
import csv
import bm25_training
import time
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# assumes all queries are from the same lang
def vectorize_query_batch(query_batch):
    manager = bm25_training.COOMatrixManager(query_batch[0].getLang())
    empty_query_indices = []
    lang = query_batch[0].getLang()

    for query in query_batch:
        tokens = Counter(query.getTokens())
        if len(tokens.values()) == 0:
            empty_query_indices.append(query.getId())
            continue

        for word, freq in tokens.items():
            manager.add(query.getId(), bm25_training.get_int(word, lang), freq)

    return manager.get_csr_matrix(), manager.get_docid_row_idx_mapping(), empty_query_indices


def process_lang(lang: str, args, queries, queue):
    bm25_scores = load_npz(f"{args.bm25_save_path}_{lang}.npz")
    docid_row_mapping = utils.load(f"{args.docid_row_mapping_save_path}_{lang}.pkl")
    bm25_scores_T = bm25_scores.T
    len_queries = len(queries)

    with ProcessPoolExecutor() as executor:
        for i in range(0, len_queries, utils.LOAD_BATCH_SIZE):
            max_idx = min(i + utils.LOAD_BATCH_SIZE, len_queries)
            query_batch = queries[i:max_idx]
            query_matrix, query_id_mapping, empty_query_indices = vectorize_query_batch(query_batch)

            docids = list(docid_row_mapping.values())
            for idx in empty_query_indices:
                queue.put([idx, random.sample(docids, 10)])
            del docids
            
            # Perform sparse matrix multiplication: [batch_query, len_vocab] * [len_vocab, num_docs]
            bm25_scores_for_queries = query_matrix.dot(bm25_scores_T)
            
            futures = [executor.submit(process_row, bm25_scores_for_queries.getrow(i)) for i in range(bm25_scores_for_queries.shape[0])]
            top_docs_per_query = [future.result() for future in futures]

            for i, top_docs in enumerate(top_docs_per_query):
                queue.put([query_id_mapping[i], [docid_row_mapping[j] for j in top_docs]])
    queue.put(-1)

def process_row(query_scores_row):
    nonzero_indices = query_scores_row.indices
    nonzero_scores = query_scores_row.data
    top_nonzero_idx = np.argsort(-nonzero_scores)[:10]
    top_docs = nonzero_indices[top_nonzero_idx]

    return top_docs




    
if __name__ == "__main__":
    start = time.time()
    args = utils.get_args()
    bm25_training.ARGS = args
    bm25_training.args = args
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

    queue = multiprocessing.Queue()
    print("Creating processes...")
    processes = []
    for lang, query_list in queries.items():
        p = multiprocessing.Process(target=process_lang, args=(lang, args, query_list, queue))
        p.start()
        processes.append(p)
    print("All processes created.")

    num_processes = len(processes)
    num_finished = 0

    with open(args.inference_output_save_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "docids"])

        while num_finished < num_processes:
            result = queue.get()
            if result == -1:
                num_finished += 1
                continue

            writer.writerow(result)
        
        for p in processes:
            p.join()

    end = time.time()
    print("Time taken: ", end - start)





        
     

        

