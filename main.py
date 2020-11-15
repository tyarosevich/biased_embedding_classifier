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

#%%
# Obtain the vectors to idenify a gender subspace

# Function to retrieve vector by word. Gensim is a little obtuse.
def get_vector(word, labels, vectors):
    idx = np.where(labels == word)
    return np.squeeze(vectors[idx, :])

gender_pair_list = [ ('she', 'he'), ('her', 'his'), ('woman', 'man'), ('Mary', 'John'), ('herself', 'himself'),
                     ('daughter', 'son'), ('mother', 'father'), ('gal', 'guy'), ('girl', 'boy'), ('female', 'male')]

gender_subspace = np.zeros( (300,10) )

for i, pair in enumerate(gender_pair_list):
    vec = get_vector(pair[0], labels, vectors) - get_vector(pair[1], labels, vectors)
    gender_subspace[:, i] = vec

#%%

norm_gsub = gender_subspace / gender_subspace.sum(axis = 0, keepdims = 1)
U, S, VT = np.linalg.svd(norm_gsub)

#%%
total = np.sum(S)
perc_sigma = S / total

plt.bar(range(10), perc_sigma)
plt.ylabel('$\sigma$ value')
plt.xlabel('$\sigma$')