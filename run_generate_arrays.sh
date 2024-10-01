#! /usr/bin/zsh
# change the shell used 

# comment this line / change it if you use conda
source ~/.zshrc
conda activate DIS-retrieval-text

python3 dict_to_tensor.py \
    --tf_idf_path ".cache/tf_idf_base.pkl" \
    --vocab_path ".cache/vocabulary.pkl" \
    --tf_idf_arrays_save_path ".cache/tf_idf_arrays_base.pkl"