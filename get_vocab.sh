#! /usr/bin/zsh

source install_dependencies.sh

python3 get_vocabulary.py \
    --stopwords True \
    --lowercase True \
    --remove_punctuation True \
    --lemmatize True \
    --idf_save_path "" \
    --tf_idf_save_path "" 
