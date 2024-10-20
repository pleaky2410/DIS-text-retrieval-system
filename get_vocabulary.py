import utils
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import gc
from collections import defaultdict
import random


def compute_vocabulary(batch):
    vocabulary = set()
    for doc in batch:
        vocabulary.update(doc.getTokens())
    return vocabulary

if __name__ == "__main__":
        args = utils.get_args()
        config = utils.args_to_doc_processing_config(args)

        num_cores = os.cpu_count() 
        divide_batch = num_cores if num_cores is not None else 16
        vocabularies = defaultdict(set)

        print("Getting Vocabulary...")
        with ProcessPoolExecutor() as executor:
            for batch in tqdm(utils.batch_load_documents(executor, divide_batch, config)):
                for doc in batch:
                    vocabularies[doc.getLang().value].update(doc.getTokens())
                    if random.random() < 0.001:
                        print("Generated tokens: ", doc.getTokens())

        utils.cleanup_all()
        gc.collect()

        for lang, vocabulary in vocabularies.items():
            print("\nLanguage: ", lang)
            print("Vocabulary Size: ", len(vocabulary))
            save_path = args.vocab_save_path + f"_{lang}.pkl"
            utils.save(save_path, vocabulary)
        

        for lang, vocabulary in vocabularies.items():
            print("\nCreating Mapping for Vocabulary for Language: ", lang)
            mapping = {}
            for i, word in tqdm(enumerate(vocabulary)):
                mapping[word] = i
            
            save_path = args.vocab_mapping_save_path + f"_{lang}.pkl"
            utils.save(save_path, mapping)