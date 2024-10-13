#! /usr/bin/zsh
# change the shell used 

source install_dependencies.sh

# python3 tf_idf_training.py \
#     --idf_save_path ".cache/idf_with_numbers" \
#     --tf_idf_save_path ".cache/tf_idf_with_numbers" \
#     --vocab_mapping_save_path ".cache/vocab_mapping_with_numbers" \
#     --docid_row_mapping_save_path ".cache/docid_row_mapping_with_numbers" \
#     --stopwords True \
#     --lowercase True \
#     --remove_punctuation True \
#     --lemmatize True \
#     --remove_numbers False \
#     --remove_special_chars True \

# python3 tf_idf_training.py \
#     --idf_save_path ".cache/idf_no_lemma" \
#     --tf_idf_save_path ".cache/tf_idf_no_lemma" \
#     --vocab_mapping_save_path ".cache/vocab_mapping_no_lemma" \
#     --docid_row_mapping_save_path ".cache/docid_row_mapping_no_lemma" \
#     --stopwords True \
#     --remove_numbers True \
#     --lowercase True \
#     --remove_punctuation True \
#     --lemmatize False \
#     --remove_special_chars True 


# python3 tf_idf_training.py \
#     --idf_save_path ".cache/idf_all_use_prob_idf" \
#     --tf_idf_save_path ".cache/tf_idf_all_use_prob_idf" \
#     --vocab_mapping_save_path ".cache/vocab_mapping_all" \
#     --docid_row_mapping_save_path ".cache/docid_row_all_use_prob_idf" \
#     --stopwords True \
#     --lowercase True \
#     --remove_punctuation True \
#     --lemmatize True \
#     --remove_numbers True \
#     --remove_special_chars True \
#     --use_prob_idf True 

python3 tf_idf_training.py \
    --idf_save_path ".cache/idf_all_use_tf_log_ave" \
    --tf_idf_save_path ".cache/tf_idf_all_use_tf_log_ave" \
    --vocab_mapping_save_path ".cache/vocab_mapping_all" \
    --docid_row_mapping_save_path ".cache/docid_row_all_use_tf_log_ave" \
    --stopwords True \
    --lowercase True \
    --remove_punctuation True \
    --lemmatize True \
    --remove_numbers True \
    --remove_special_chars True \
    --use_tf_log_ave True 

python3 tf_idf_training.py \
    --idf_save_path ".cache/idf_all_use_tf_augmented" \
    --tf_idf_save_path ".cache/tf_idf_all_use_tf_augmentend" \
    --vocab_mapping_save_path ".cache/vocab_mapping_all" \
    --docid_row_mapping_save_path ".cache/docid_row_all_use_tf_augmented" \
    --stopwords True \
    --lowercase True \
    --remove_punctuation True \
    --lemmatize True \
    --remove_numbers True \
    --remove_special_chars True \
    --use_tf_augmented True 