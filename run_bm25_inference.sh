#! /usr/bin/zsh
# change the shell used 

source install_dependencies.sh


python3 bm25_inference.py \
    --bm25_save_path ".cache/bm25_all_bm25l" \
    --docid_row_mapping_save_path ".cache/docid_row_bm25_all_bm25l" \
    --vocab_mapping_save_path ".cache/vocab_mapping_all" \
    --stopwords True \
    --lowercase True \
    --lemmatize True \
    --remove_punctuation True \
    --remove_special_chars True \
    --inference_output_save_path "bm25l.csv" \
    --remove_numbers True

python3 bm25_inference.py \
    --bm25_save_path ".cache/bm25_all_tf_nonlinear" \
    --docid_row_mapping_save_path ".cache/docid_row_bm25_all_tf_nonlinear" \
    --vocab_mapping_save_path ".cache/vocab_mapping_all" \
    --stopwords True \
    --lowercase True \
    --lemmatize True \
    --remove_punctuation True \
    --remove_special_chars True \
    --inference_output_save_path "tf_nonlinear.csv" \
    --remove_numbers True

python3 bm25_inference.py \
    --bm25_save_path ".cache/bm25_all_atire" \
    --docid_row_mapping_save_path ".cache/docid_row_bm25_all_atire" \
    --vocab_mapping_save_path ".cache/vocab_mapping_all" \
    --stopwords True \
    --lowercase True \
    --lemmatize True \
    --remove_punctuation True \
    --remove_special_chars True \
    --inference_output_save_path "atire.csv" \
    --remove_numbers True