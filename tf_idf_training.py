from collections import defaultdict
import os
import numpy as np
from utils import Document 
from concurrent.futures import ProcessPoolExecutor
import utils
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import gc
from scipy.sparse import coo_matrix, save_npz

def compute__basic_tf(doc: Document)-> tuple[int, utils.Lang, dict]:
    """
    Compute the term frequency (TF) for each word in a document.

    Parameters:
    doc (str): The input document as an object Document.

    Returns:
    dict: A dictionary where keys are words and values are their normalized term frequencies.
    """
    words = doc.getTokens()
    if len(words) == 0:
        return doc.getId(), doc.getLang(), {}

    tf = Counter(words)
    lang = doc.getLang()

   # normalize by the maximum term frequency
    max_value = tf.most_common(1)[0][1]

    tf = {get_int(word, lang): freq / max_value for word, freq in tf.items()}

    return doc.getId(), lang, tf


def compute_boolean_tf(doc: Document):
    id_doc, lang, tf = compute__basic_tf(doc)
    return id_doc, lang, {word: 1 if tf[word] > 0 else 0 for word in tf}


def compute_logarithm_tf(doc: Document):
    id_doc, lang, tf = compute__basic_tf(doc)
    return id_doc, lang, {word: 1 + np.log(tf[word]) if tf[word] > 0 else 1 + np.log(1e-10) for word in tf}


def compute_augmented_tf(doc: Document):
    id_doc, lang, tf = compute__basic_tf(doc)
    if len(tf) == 0:
        return id_doc, lang, tf
    max_tf = max(tf.values())
    max_tf = max(max_tf, 1e-10)
    return id_doc, lang, {word: 0.5 + 0.5 * tf[word] / max_tf for word in tf}


def compute_log_average_tf(doc: Document):
    id_doc, lang, tf = compute_logarithm_tf(doc)
    if len(tf) == 0:
        return id_doc, lang, tf
    avg_tf = sum(tf.values()) / len(tf)
    avg_tf = max(avg_tf, 1e-10)
    return id_doc, lang, { word: tf[word] / (1 + np.log(avg_tf)) for word in tf}


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

        
def compute_lang_tf_idf(args, divide_batch, lang):
    global mappings, mappings_save_path
    idf = np.zeros(get_vocabulary_size(lang) + 1)
    mappings_save_path = args.vocab_mapping_save_path
    print(f"Language: {lang.value}\tIdf shape: {idf.shape}")
    

    config = utils.args_to_doc_processing_config(args)

    tf_idf_sparse = COOMatrixManager(lang)
    tf_idf = defaultdict(dict)
    total_docs = 0

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for batch in tqdm(utils.batch_load_documents(executor=executor, divide_batch=divide_batch, config=config, lang=lang)):
            total_docs += len(batch)
            
            list_unique_words_per_lang =  compute_unique_words(batch)

            assert(len(list_unique_words_per_lang) == 1) #safety check

            for _, list_unique_words in list_unique_words_per_lang.items():
                for doc_unique_words in list_unique_words:
                    for word in doc_unique_words:
                        idf[get_int(word, lang)] += 1
            
                if args.use_tf_log_ave:
                    tfs_docs_id = list(executor.map(compute_log_average_tf, batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))
                elif args.use_tf_augmented:
                    tfs_docs_id = list(executor.map(compute_augmented_tf, batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))
                elif args.use_tf_boolean:
                    tfs_docs_id = list(executor.map(compute_boolean_tf, batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))
                elif args.use_tf_log:
                    tfs_docs_id = list(executor.map(compute_logarithm_tf, batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))
                else:
                    tfs_docs_id = list(executor.map(compute__basic_tf, batch, chunksize=utils.LOAD_BATCH_SIZE // divide_batch))

                for doc_id, _ , tf_doc in tfs_docs_id:
                    for int_word, tf in tf_doc.items():
                        tf_idf[int_word][doc_id] = tf

                del tfs_docs_id
                gc.collect()
                print(f"Processed {total_docs} documents", end="\r")
            

    del config
    gc.collect()

    print("Calculating idf and tf-idf scores...")

    for i, count in tqdm(enumerate(idf), mininterval=7):
        if args.use_prob_idf:
            if count == 0:
                idf[i] = 0
            else:
                idf[i] = max(0, np.log((total_docs - count) / count))
        else:
            idf[i] = np.log(total_docs / (1 + count))
    
    for int_word, docs_tf_idf in tf_idf.items():
        for doc_id, tf_val in docs_tf_idf.items():
            tf_idf_sparse.add(doc_id, int_word, tf_val * idf[int_word])
    
    mappings = {}
    del total_docs
    del divide_batch
    del tf_idf
    gc.collect()

    print("Saving idf scores...")
    np.save(f"{args.idf_save_path}_{lang.value}.npy", idf)

    tf_idf_save_path = args.tf_idf_save_path
    docid_row_idx_mapping_save_path = args.docid_row_mapping_save_path
    del args
    del idf
    gc.collect()

    print("Saving tf-idf scores...")
    save_npz(f"{tf_idf_save_path}_{lang.value}.npz", tf_idf_sparse.get_csr_matrix())
    utils.save(f"{docid_row_idx_mapping_save_path}_{lang.value}.pkl", tf_idf_sparse.get_docid_row_idx_mapping())

        

if __name__ == "__main__"  :
    args = utils.get_args()
    ARGS = args

    num_cores = os.cpu_count() 
    divide_batch = num_cores if num_cores is not None else 16

    if args.vocab_mapping_save_path is None:
        exit("Please provide a path to load the vocabulary mapping.")

    for lang in utils.Lang:
        compute_lang_tf_idf(args, divide_batch, lang)

