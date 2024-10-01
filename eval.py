

def set_intersect(expected_doc_ids: set, retrieved_doc_ids: set) -> int :
    return len(expected_doc_ids & retrieved_doc_ids)


def precision_at_k(expected_doc_ids: set, retrieved_doc_ids: list, k=None) -> float:
    """
    Compute the precision at k.
    
    Parameters:
    expected_doc_ids (set): A set of document IDs that are relevant.
    retrieved_doc_ids (list): A list of document IDs that were retrieved.
    k (int): The number of documents to consider.
    
    Returns:
    float: The precision at k.
    """
    if k is None:
        k = len(retrieved_doc_ids)
    else:
        k = min(k, len(retrieved_doc_ids))

    retrieved_doc_ids_set = set(retrieved_doc_ids[:k])
    
    return set_intersect(expected_doc_ids, retrieved_doc_ids_set) / k

def recall_at_k(expected_doc_ids: set, retrieved_doc_ids: list, k=None) -> float:
    """
    Compute the recall at k.
    
    Parameters:
    expected_doc_ids (set): A set of document IDs that are relevant.
    retrieved_doc_ids (list): A list of document IDs that were retrieved.
    k (int): The number of documents to consider.
    
    Returns:
    float: The recall at k.
    """
    if k is None:
        k = len(retrieved_doc_ids)
    else:
        k = min(k, len(retrieved_doc_ids))

    retrieved_doc_ids_set = set(retrieved_doc_ids[:k])
    
    return set_intersect(expected_doc_ids, retrieved_doc_ids_set) / len(expected_doc_ids)


def mean_average_precision(expected_doc_ids: list, retrieved_doc_ids: list)
   """
   Compute the mean average precision of a set (possibly of size 1) of queries.

    Parameters:
    expected_doc_ids (set): A list of sets of document IDs that are relevant for each query

    retrieved_doc_ids (list): A list of lists of document IDs that were retrieved for each query

    Returns:
    float: The mean average precision.
   """ 

   
   if len(expected_doc_ids) != len(retrieved_doc_ids):
       raise ValueError("The number of queries must be the same in both lists.")
    
    if len(expected_doc_ids) == 0:
         return 0
    
   if len(expected_doc_ids) == 1:
       return average_precision(expected_doc_ids[0], retrieved_doc_ids[0])
    
    ap = [average_precision(expected, retrieved) for expected, retrieved in zip(expected_doc_ids, retrieved_doc_ids)]

    return sum(ap) / len(ap)




def average_precision(expected_doc_ids: set, retrieved_doc_ids: list) -> float:
    res = 0
    total_relevant_retrieved = 0
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in expected_doc_ids:
            total_relevant_retrieved += 1
            res += precision_at_k(expected_doc_ids, retrieved_doc_ids, i)

    return res / total_relevant_retrieved if total_relevant_retrieved > 0 else 0
