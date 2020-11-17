#### NOTE THIS FILE INCLUDES RE-IMPLEMENTATION OF SMALL AMOUNTS OF CODE
#### FROM THE ORIGINAL PAPER FOUND AT: https://github.com/tolga-b/debiaswe/blob/master/debiaswe/debias.py

import gensim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Various functions to process the corpus.

# Some functions for playing

# Displays the n most similar words to a given word.
def n_most_similar(model, word, n):
    list = model.most_similar(positive = word, topn = n)
    return list

# Displays the cosine similarity between two words.
def compare_words(model, word1, word2):
    result = model.similarity(word1, word2)
    return result

# Returns a list of tuples of the most similar word to w1 + w2 - w3 and whatever the cos sim is.
def x_y_minusz(model, w1, w2, w3, n):
    result = model.most_similar(positive = [w1, w2], negative = [w3], topn = n)
    return result

# A sanity check function to confirm that the numpy array of the corpus
# Corresponds to the gensim model
def confirm_key(n,model, vectors, labels):
    '''
    Parameters
    ----------
    n: int
        An arbitrary word index between 0 and 2,999,999

    Returns
    -------
    None
    '''
    test_word = labels[n]
    test_key = vectors[n, :]
    model_key = model.get_vector(test_word)
    print("For test word '%s' the comparison of the model and numpy array vectors is: %r." % (test_word, np.array_equal(test_key, model_key)))

# Function to retrieve vector by word. Gensim is a little obtuse.
def get_vector(word, labels, vectors):
    '''

    Parameters
    ----------
    word: str
        The word to look up.
    labels: list
        List of the word labels for the embeddings.
    vectors: ndarray
        Corpus of word embeddings

    Returns
    -------
    ndarray
        Word embedding vector for the word.
    '''
    idx = np.where(labels == word)
    return np.squeeze(vectors[idx, :])


# Returns a subspace based on a list of pairs that are assumed
# to have a meaningful analogy relation such as gender or racial pairs.
def get_subspace(labels, vectors, pair_list):
    '''
    Parameters
    ----------
    labels: list
        List of labels for embeddings
    vectors: ndarray
        Corpus of word embeddings
    pair_list: list
        List of tuples of analogy pairs.

    Returns
    -------
    ndarray
        Numpy array of concatenated vectors of difference between pairs.
    '''
    dim = (np.shape(vectors)[1], len(pair_list))
    subspace = np.zeros(dim)

    for i, pair in enumerate(pair_list):
        vec = get_vector(pair[0], labels, vectors) - get_vector(pair[1], labels, vectors)
        subspace[:, i] = vec
    return subspace

# Converts columns of the subspace to unit vectors,
# and returns the SVD.
def norm_svd(subspace):
    '''
    Parameters
    ----------
    subspace: ndarray
        The matrix representing the subspace to be studied.

    Returns
    -------
    tuple
        A tuple containing the SVD of the normalized subspace. Format is
        (U, S, VT) where VT signifies transpose of V.
    '''
    norm_gsub = subspace / subspace.sum(axis = 0, keepdims = 1)
    U, S, VT = np.linalg.svd(norm_gsub)
    return (U, S, VT)

# Plots the singular values as a % of total variance.
def sing_value_plot(S, **kwargs):
    '''
    Parameters
    ----------
    S: ndarray
        The vector containing the singular values.

    Returns
    -------
    None

    '''
    total = np.sum(S)
    perc_sigma = S / total

    ax = plt.bar(range(10), perc_sigma, **kwargs)
    plt.ylabel('$\sigma$ value')
    plt.xlabel('$\sigma$')
    return ax

def get_PCA_subspace(pair_list, labels, embeddings, num_sigmas = 10):
    '''
    Parameters
    ----------
    pair_list: list
        List of gender word pair tuples
    labels: list
        list of string labels associated with embeddings
    embeddings: ndarray
        corpus of word2vec embeddings
    num_sigmas: int
        Number of components to use in the PCA

    Returns
    -------
    ndarray
    '''
    dim = (np.shape(embeddings)[1], len(pair_list) * 2)
    matrix = np.empty(dim)
    for i, pair in enumerate(pair_list):
        v1 = get_vector(pair[0], labels, embeddings)
        v2 = get_vector(pair[1], labels, embeddings)
        center = (v1 + v2) / 2
        matrix[:, i * 2] = v1 - center
        matrix[:, i*2 + 1] = v2 - center
    pca = PCA(n_components = num_sigmas)
    pca.fit(matrix)
    return matrix