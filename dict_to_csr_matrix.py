import utils
from tf_idf_training import get_coo_matrix_manager
import tf_idf_training
from scipy.sparse import save_npz

args = utils.get_args()
tf_idf_training.ARGS = args
for lang in utils.Lang:
    print(f'lang: {lang}')
    tf_idf = utils.load(f".cache/tf_idf_test_{lang.value}.pkl")

    manager = get_coo_matrix_manager(tf_idf, lang)
    csrmatrix = manager.get_csr_matrix()
    mapping = manager.get_docid_row_idx_mapping()

    print(csrmatrix.shape)
    print(len(mapping))

    save_npz(f".cache/csrmatrix_test_{lang.value}.npz", csrmatrix)
    utils.save(f".cache/doc_id_row_mapping_test_{lang.value}.pkl", mapping)