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
    --idf_save_path ".cache/idf_use_prob_idf_with_numbers" \
    --tf_idf_save_path ".cache/tf_idf_use_prob_idf_with_numbers" \
    --docid_row_mapping_save_path ".cache/docid_row_use_prob_idf_with_numbers" \
    --vocab_mapping_save_path ".cache/vocab_mapping_with_numbers" \
    --stopwords True \
    --lowercase True \
    --lemmatize True \
    --remove_punctuation True \
    --remove_special_chars True \
    --use_prob_idf True \
    --inference_output_save_path "output_use_prob_idf_with_numbers.csv"
    # --remove_numbers True

# python3 tf_idf_inference.py \
#     --idf_save_path ".cache/idf_all_use_tf_log_ave" \
#     --tf_idf_save_path ".cache/tf_idf_all_use_tf_log_ave" \
#     --docid_row_mapping_save_path ".cache/docid_row_all_use_tf_log_ave" \
#     --vocab_mapping_save_path ".cache/vocab_mapping_all" \
#     --stopwords True \
#     --lowercase True \
#     --lemmatize True \
#     --remove_punctuation True \
#     --remove_special_chars True \
#     --inference_output_save_path "output_use_tf_log_ave.csv" \
#     --use_tf_log_ave True \
#     --remove_numbers True

# python3 tf_idf_inference.py \
#     --idf_save_path ".cache/idf_all_use_tf_augmented" \
#     --tf_idf_save_path ".cache/tf_idf_all_use_tf_augmentend" \
#     --docid_row_mapping_save_path ".cache/docid_row_all_use_tf_augmented" \
#     --vocab_mapping_save_path ".cache/vocab_mapping_all" \
#     --stopwords True \
#     --lowercase True \
#     --lemmatize True \
#     --remove_punctuation True \
#     --remove_special_chars True \
#     --inference_output_save_path "output_use_tf_augmented.csv" \
#     --use_tf_augmented True \
#     --remove_numbers True 