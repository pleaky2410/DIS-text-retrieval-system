import math
from collections import defaultdict
import pickle
import os

def compute_tf(doc):
    """
    Compute the term frequency (TF) for each word in a document.

    Parameters:
    doc (str): The input document as a string.

    Returns:
    dict: A dictionary where keys are words and values are their normalized term frequencies.
    """
    tf = defaultdict(float)
    words = doc.split()
    total_words = len(words)

    # Count occurrences of each word
    for word in words:
        tf[word] += 1
    
    # Normalize by total word count
    for word in tf:
        tf[word] /= total_words

    return tf

def compute_idf(documents):
    """
    Compute the inverse document frequency (IDF) for each unique word across all documents.

    Parameters:
    documents (list): A list of documents (strings).

    Returns:
    dict: A dictionary where keys are words and values are their IDF scores.
    """
    idf = defaultdict(float)
    total_docs = len(documents)

    # Count unique occurrences of words in each document
    for doc in documents:
        unique_words = set(doc.split())
        for word in unique_words:
            idf[word] += 1

    # Calculate IDF for each word
    for word in idf:
        idf[word] = math.log(total_docs / (1 + idf[word]))  # Adding 1 to avoid division by zero

    return idf

def compute_tfidf(documents, save_path="./.cache/tf-idf.scores"):
    """
    Compute the TF-IDF scores for each document in a list of documents.

    Parameters:
    documents (list): A list of documents (strings).

    Returns:
    list: A list of dictionaries, each containing TF-IDF scores for the corresponding document.
    """
    tfidf = []
    idf = compute_idf(documents)

    for doc in documents:
        tf = compute_tf(doc)
        doc_tfidf = {word: tf[word] * idf[word] for word in tf}
        tfidf.append(doc_tfidf)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(tfidf, f)

    return tfidf

# Example usage
if __name__ == "__main__":
    documents = [
        "the cat in the hat",
        "the quick brown fox",
        "the cat sat on the mat"
    ]

    tfidf_results = compute_tfidf(documents)
    for i, doc in enumerate(tfidf_results):
        print(f"Document {i + 1} TF-IDF: {doc}")