from collections import defaultdict
import os
import numpy as np
from utils import Document , Query
from concurrent.futures import ProcessPoolExecutor
import utils
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import gc
from scipy.sparse import coo_matrix, save_npz
from functools import partial
import multiprocessing

def compute__basic_tf(doc: Document | Query)-> tuple[int, int, dict]:
    """
    Compute the term frequency (TF) for each word in a document.

    Parameters:
    doc (str): The input document as an object Document.

    Returns:
    dict: A dictionary where keys are words and values are their normalized term frequencies.
    """
    words = doc.getTokens()
    doc_len = len(words)
    if doc_len == 0:
        return doc.getId(), 0, {}

    tf = Counter(words)
    lang = doc.getLang()
    return doc.getId(), doc_len, {get_int(word, lang): count for word, count in tf.items()}


def compute_boolean_tf(doc: Document | Query):
    id_doc, lang, tf = compute__basic_tf(doc)
    return id_doc, lang, {word: 1 if tf[word] > 0 else 0 for word in tf}


def compute_logarithm_tf(doc: Document | Query):
    id_doc, lang, tf = compute__basic_tf(doc)
    return id_doc, lang, {word: 1 + np.log(tf[word]) if tf[word] > 0 else 0 for word in tf}


def compute_augmented_tf(doc: Document | Query):
    id_doc, lang, tf = compute__basic_tf(doc)
    if len(tf) == 0:
        return id_doc, lang, tf
    max_tf = max(tf.values())
    return id_doc, lang, {word: 0.5 + 0.5 * tf[word] / max_tf if max_tf != 0 else 0.5 for word in tf}


def compute_log_average_tf(doc: Document | Query):
    id_doc, lang, tf = compute_logarithm_tf(doc)
    if len(tf) == 0:
        return id_doc, lang, tf
    avg_tf = sum(tf.values()) / len(tf)
    return id_doc, lang, { word: tf[word] / (1 + np.log(avg_tf)) if avg_tf != 0 else tf[word] for word in tf}


def compute_normalization_pivot_tf(el : float, avg_number_distinct_words_in_doc: float, number_distinct_terms_in_doc: int):
    s = 0.2
    return el / ((1-s)*avg_number_distinct_words_in_doc + s*number_distinct_terms_in_doc)


def compute_unique_words(documents: list[Document]):
    """
    Compute the set of unique words across a batch of documents.

    Parameters:
    documents (list): A list of documents (strings).

    Returns:
    list: A list of sets containing all unique words in the documents.
    """

    unique_words = defaultdict(list)
    for doc in documents:
        unique_words[doc.getLang()].append(set(doc.getTokens()))
    return unique_words


mappings = {}
mappings_save_path = None
num_langs = len(utils.Lang)
def alleviate():
    global mappings

    if len(mappings) >=   num_langs // 2:
        if 'ko' in mappings:
            del mappings['ko']
        if 'it' in mappings:
            del mappings['it']
        if 'ar' in mappings:
            del mappings['ar']
        if 'es' in mappings:
            del mappings['es']
        if 'de' in mappings:
            del mappings['de']
        gc.collect()
        if len(mappings) >= num_langs // 2:
            if 'en' in mappings:
                del mappings['en']
            gc.collect()

def get_int(word: str, lang: utils.Lang):
    global mappings
    if lang.value not in mappings:
        save_path = args.vocab_mapping_save_path + f"_{lang.value}.pkl"
        mappings[lang.value] = utils.load(save_path)

    mapping = mappings[lang.value]
    if word not in mapping:
        alleviate()
        return len(mapping)

    res = mapping[word]
    alleviate()
    return res

def get_vocabulary_size(lang: utils.Lang):
    if lang.value not in mappings:
        save_path = ARGS.vocab_mapping_save_path + f"_{lang.value}.pkl"
        mappings[lang.value] = utils.load(save_path)
    res = len(mappings[lang.value])
    alleviate()
    return res
    
class COOMatrixManager:
    def __init__(self, lang):
        self.lang = lang    
        self.data = []
        self.row = []
        self.col = []
        self.docid_row_idx_mapping = {}
    
    def add(self, docid, int_word, value):
        if docid not in self.docid_row_idx_mapping:
            self.docid_row_idx_mapping[docid] = len(self.docid_row_idx_mapping)
        self.data.append(value)
        self.row.append(self.docid_row_idx_mapping[docid])
        self.col.append(int_word)
    

    def get_csr_matrix(self):
        if hasattr(self, "csr_matrix"):
            return self.csr_matrix

        self.csr_matrix = coo_matrix((self.data, (self.row, self.col)), shape=(len(self.docid_row_idx_mapping), get_vocabulary_size(self.lang) + 1)).tocsr()
        del self.data
        del self.row
        del self.col

        return self.csr_matrix
    
    def get_docid_row_idx_mapping(self):
        res = {}
        for docid, idx in self.docid_row_idx_mapping.items():
            res[idx] = docid
        return res

def get_coo_matrix_manager(tf_idf, lang):
    manager = COOMatrixManager(lang)
    for int_word, docs_tf_idf in tf_idf.items():
        for doc_id, tf_val in docs_tf_idf.items():
            manager.add(doc_id, int_word, tf_val)
    
    return manager

        
def compute_lang_bm25(args, divide_batch, lang):
    global mappings, mappings_save_path
    idf = np.zeros(get_vocabulary_size(lang) + 1)
    mappings_save_path = args.vocab_mapping_save_path
    print(f"Language: {lang.value}\tIdf shape: {idf.shape}")
    

    config = utils.args_to_doc_processing_config(args)

    bm25_sparse = COOMatrixManager(lang)
    bm25 = defaultdict(dict)
    total_docs = 0
    avg_doc_len = 0

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for batch in tqdm(utils.batch_load_documents(executor=executor, divide_batch=divide_batch, config=config, lang=lang)):
            total_docs += len(batch)
            
            list_unique_words_per_lang =  compute_unique_words(batch)
            avg_doc_len += sum(len(doc.getTokens()) for doc in batch)

            assert(len(list_unique_words_per_lang) == 1) #safety check

            for _, list_unique_words in list_unique_words_per_lang.items():
                for doc_unique_words in list_unique_words:
                    for word in doc_unique_words:
                        idf[get_int(word, lang)] += 1
            
                tfs_docs_id = list(executor.map(compute__basic_tf, batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))

                for doc_id, doc_len , tf_doc in tfs_docs_id:
                    for int_word, count in tf_doc.items():
                        bm25[int_word][doc_id] = count, doc_len

                del tfs_docs_id
                gc.collect()
                print(f"Processed {total_docs} documents", end="\r")
            

    del config
    gc.collect()

    print("Calculating idf and tf-idf scores...")

    avg_doc_len /= total_docs

    for i, count in tqdm(enumerate(idf), mininterval=7):
        if args.variant == "atire":
            idf[i] = np.log(total_docs / count)
        
        elif args.variant == "bm25l":
            idf[i] = np.log((total_docs + 1) / (count + 0.5))
        
        elif args.variant == "bm25+" or args.variant == "tf_nonlinear":
            idf[i] = np.log((total_docs + 1) / count)

        elif args.variant == "lucene":
            idf[i] = np.log((total_docs - count + 0.5) / (count + 0.5) + 1) 
    
    for int_word, docs_tf_idf in bm25.items():
        for doc_id, (count, doc_len) in docs_tf_idf.items():
            if args.variant == "atire":
                tf = (args.k_1 + 1) * count / (args.k_1 * (1 - args.b + args.b * doc_len / avg_doc_len) + count)

            elif args.variant == "bm25l":
                c_td = count / (1 - args.b + args.b * doc_len / avg_doc_len)
                tf = (args.k_1 + 1) * (c_td * args.gamma) / (args.k_1 + c_td + args.gamma)

            elif args.variant == "bm25+":
                tf = (args.k_1 + count) / (args.k_1 * (1 - args.b + args.b * doc_len / avg_doc_len) + count) + args.gamma

            elif args.variant == "lucene":
                tf = count  / (count + args.k_1 * (1 - args.b + args.b * doc_len / avg_doc_len))
            
            elif args.variant == "tf_nonlinear":
                tf = 1 + np.log( 1 + np.log(count / (1 - args.b + args.b * doc_len / avg_doc_len) + args.gamma))

            bm25_sparse.add(doc_id, int_word, tf * idf[int_word])
    
    mappings = {}
    del total_docs
    del divide_batch
    del bm25
    gc.collect()

    print("Saving bm25 scores...")
    save_npz(f"{args.bm25_save_path}_{lang.value}.npz", bm25_sparse.get_csr_matrix())
    utils.save(f"{args.docid_row_mapping_save_path}_{lang.value}.pkl", bm25_sparse.get_docid_row_idx_mapping())

        

if __name__ == "__main__"  :
    args = utils.get_args()
    ARGS = args

    num_cores = os.cpu_count() 
    divide_batch = num_cores if num_cores is not None else 16

    if args.vocab_mapping_save_path is None:
        exit("Please provide a path to load the vocabulary mapping.")

    langs = [lang for lang in utils.Lang]
    for i in range(0, len(langs), 3):
        max_idx = min(i+3, len(langs))
        lang_batch = langs[i:max_idx]
        processes = []
        for lang in lang_batch:
            p = multiprocessing.Process(target=compute_lang_bm25, args=(args, divide_batch, lang))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()

