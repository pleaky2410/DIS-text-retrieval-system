#! /usr/bin/zsh
# change the shell used 

source install_dependencies.sh

# python3 tf_idf_inference.py \
#     --idf_save_path ".cache/idf_test" \
#     --tf_idf_save_path ".cache/tf_idf_all" \
#     --docid_row_mapping_save_path ".cache/docid_row_mapping_all" \
#     --vocab_mapping_save_path ".cache/vocab_mapping_all" \
#     --stopwords True \
#     --lowercase True \
#     --lemmatize True \
#     --remove_punctuation True \
#     --remove_special_chars True \
#     --remove_numbers True 


python3 tf_idf_inference.py \
    --idf_save_path ".cache/idf_test" \
    --tf_idf_save_path ".cache/tf_idf_all" \
    --docid_row_mapping_save_path ".cache/docid_row_mapping_all" \
    --vocab_mapping_save_path ".cache/vocab_mapping_all" \
    --stopwords True \
    --lowercase True \
    --lemmatize True \
    --remove_punctuation True \
    --remove_special_chars True \
    --remove_numbers True 
