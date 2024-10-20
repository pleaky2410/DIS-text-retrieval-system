#! /usr/bin/zsh

source install_dependencies.sh

# python3 get_vocabulary.py \
#     --stopwords True \
#     --lowercase True \
#     --remove_punctuation True \
#     --remove_special_chars True \
#     --lemmatize True \
#     --remove_numbers False \
#     --idf_save_path "" \
#     --tf_idf_save_path "" \
#     --vocab_save_path ".cache/vocab_with_numbers" \
#     --vocab_mapping_save_path ".cache/vocab_mapping_with_numbers" \


# python3 get_vocabulary.py \
#     --stopwords True \
#     --lowercase True \
#     --remove_punctuation True \
#     --remove_special_chars True \
#     --lemmatize False \
#     --remove_numbers True \
#     --idf_save_path "" \
#     --tf_idf_save_path "" \
#     --vocab_save_path ".cache/vocab_no_lemma" \
#     --vocab_mapping_save_path ".cache/vocab_mapping_no_lemma" \

python3 get_vocabulary.py \
    --stopwords True \
    --lowercase True \
    --remove_punctuation True \
    --remove_special_chars True \
    --lemmatize True \
    --remove_numbers True \
    --idf_save_path "" \
    --tf_idf_save_path "" \
    --vocab_save_path ".cache/vocab_all_2" \
    --vocab_mapping_save_path ".cache/vocab_mapping_all_2" 