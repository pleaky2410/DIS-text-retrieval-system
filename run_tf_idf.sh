#! /usr/bin/zsh
# change the shell used 

# comment this line / change it if you use conda
source install_dependencies.sh

    # --vocab_path ".cache/vocabulary.pkl" \
python3 tf_idf.py \
    --idf_save_path ".cache/idf_base.pkl" \
    --tf_idf_save_path ".cache/tf_idf_base.pkl" \
    --stopwords True \
    --lowercase True \
    --remove_punctuation True \
    --lemmatize True


