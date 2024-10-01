import utils
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import gc


def compute_vocabulary(batch):
    vocabulary = set()
    for doc in batch:
        vocabulary.update(doc.getTokens())
    return vocabulary

if __name__ == "__main__":
        args = utils.get_args()

        num_cores = os.cpu_count() 
        divide_batch = num_cores if num_cores is not None else 16
        vocabulary = set()

        print("Getting Vocabulary...")
        with ProcessPoolExecutor() as executor:
            for batch in tqdm(utils.batch_load_documents(executor, divide_batch, args)):
                vocabulary.update(compute_vocabulary(batch))
        utils.cleanup_all()
        gc.collect()
        
        print("Saving Vocabulary...")
        utils.save(args.vocab_save_path, vocabulary)

        print("Creating Mapping for Vocabulary...")
        mapping = {}
        for i, word in tqdm(enumerate(vocabulary)):
            mapping[word] = i

        del vocabulary
        gc.collect()

        print("Saving Mapping...")
        utils.save(args.vocab_mapping_save_path, mapping)