#! /usr/bin/zsh
# change the shell used 

source install_dependencies.sh

python3 bm25_training.py \
    --bm25_save_path ".cache/bm25_all_bm25l" \
    --vocab_mapping_save_path ".cache/vocab_mapping_all" \
    --docid_row_mapping_save_path ".cache/docid_row_bm25_all_bm25l" \
    --stopwords True \
    --lowercase True \
    --remove_punctuation True \
    --lemmatize True \
    --remove_numbers True \
    --variant "bm25l" \
    --remove_special_chars True

python3 bm25_training.py \
    --bm25_save_path ".cache/bm25_all_tf_nonlinear" \
    --vocab_mapping_save_path ".cache/vocab_mapping_all" \
    --docid_row_mapping_save_path ".cache/docid_row_bm25_all_tf_nonlinear" \
    --stopwords True \
    --lowercase True \
    --remove_punctuation True \
    --lemmatize True \
    --remove_numbers True \
    --variant "tf_nonlinear" \
    --remove_special_chars True