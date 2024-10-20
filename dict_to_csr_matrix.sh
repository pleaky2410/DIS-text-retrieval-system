#! /usr/bin/zsh

source ~/.zshrc
conda activate DIS-retrieval-text

python dict_to_csr_matrix.py \
    --idf_save_path ".cache/idf_test" \
    --tf_idf_save_path ".cache/tf_idf_test" \
    --stopwords True \
    --lowercase True \
    --remove_punctuation True \
    --vocab_mapping_save_path ".cache/vocab_mapping_test" \
    --lemmatize True \
    --remove_special_chars True \
    --remove_numbers True \
    --per_lang True 
