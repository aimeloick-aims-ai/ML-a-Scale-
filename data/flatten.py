import numpy as np
def flatten_data(data):
    """
    data_by_user : list of length n_users; each element is list of (movie_idx, rating)
    Retourne movie_idx_all, rating_all, user_starts (len = n_users+1)
    """
    n_element = len(data)
    counts = [len(x) for x in data]
    total = sum(counts)

    element_idx = np.empty(total, dtype=np.int32)
    ratings = np.empty(total, dtype=np.float32)
    starts = np.empty(n_element + 1, dtype=np.int32)

    pos = 0
    for u in range(n_element):
        starts[u] = pos
        for (m, r) in data[u]:
            element_idx[pos] = int(m)
            ratings[pos] = float(r)
            pos += 1
    starts[n_element] = pos
    return element_idx, ratings, starts


def flatten_list_of_lists(list_of_lists):
    """
    list_of_lists : liste de listes d'entiers
    Retourne data_all, starts (len = len(list_of_lists)+1)
    """
    n_element = len(list_of_lists)
    counts = [len(x) for x in list_of_lists]
    total = sum(counts)

    data_all = np.empty(total, dtype=np.int32)
    starts = np.empty(n_element + 1, dtype=np.int32)

    pos = 0
    for i in range(n_element):
        starts[i] = pos
        for item in list_of_lists[i]:
            data_all[pos] = int(item)
            pos += 1
    starts[n_element] = pos

    return data_all, starts