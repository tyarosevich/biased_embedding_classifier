#%%

import gensim
import numpy as np
from sys import getsizeof
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from matplotlib import colors
#%%
model = gensim.models.KeyedVectors.load_word2vec_format('/home/tyarosevich/code_work/word2vec_news/GoogleNews-vectors-negative300.bin', binary=True)

vocab = model.vocab.keys()
wordsInVocab = len(vocab)
print (wordsInVocab)
#%%

# See if we are working:

print (model.similarity('dog', 'puppy'))
print (model.similarity('post', 'book'))

#%%

# Some functions for playing

# Displays the n most similar words to a given word.
def n_most_similar(word, n):
    list = model.most_similar(positive = word, topn = n)
    return list

# Displays the cosine similarity between two words.
def compare_words(word1, word2):
    result = model.similarity(word1, word2)
    return result

# Returns a list of tuples of the most similar word to w1 + w2 - w3 and whatever the cos sim is.
def x_y_minusz(w1, w2, w3, n):
    result = model.most_similar(positive = [w1, w2], negative = [w3], topn = n)
    return result

#%%

print(x_y_minusz('woman', 'king', 'man', 1))

#%%

vectors = np.asarray(model.wv.vectors)
labels = np.asarray(model.wv.index2word)

#%%

def confirm_key(n):
    test_word = labels[n]
    test_key = vectors[n, :]
    model_key = model.get_vector(test_word)
    print("For test word '%s' the comparison of the model and numpy array vectors is: %r." % (test_word, np.array_equal(test_key, model_key)))


#%%
with open('w2v_as_np.pickle', 'wb') as f:
    pickle.dump(vectors, f)

with open('labels_for_npmat.pickle', 'wb') as f:
    pickle.dump(labels, f)
#%%
with open('w2v_as_np.pickle', 'rb') as f:
    vectors = pickle.load(f)

with open('labels_for_npmat.pickle', 'rb') as f:
    labels = pickle.load(f)

#%% Various functions to process the corpus.


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
def sing_value_plot(S):
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

    plt.bar(range(10), perc_sigma)
    plt.ylabel('$\sigma$ value')
    plt.xlabel('$\sigma$')
#%%
# Get the gender subspace
gender_pair_list = [ ('she', 'he'), ('her', 'his'), ('woman', 'man'), ('Mary', 'John'), ('herself', 'himself'),
                     ('daughter', 'son'), ('mother', 'father'), ('gal', 'guy'), ('girl', 'boy'), ('female', 'male')]

gender_subspace = get_subspace(labels, vectors, gender_pair_list)
U, S, VT = norm_svd(gender_subspace)
sing_value_plot(S)

