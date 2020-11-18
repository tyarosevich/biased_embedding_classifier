#%%

import gensim
import numpy as np
from sys import getsizeof
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from matplotlib import colors
import helpers
from importlib import reload
import timeit
from sklearn.decomposition import PCA


# reload helper functions because ipython is lame.
#%% Loads the word2vec news corpus into gensim to make use of gensims
# analytical tools.

model = gensim.models.KeyedVectors.load_word2vec_format('/home/tyarosevich/code_work/word2vec_news/GoogleNews-vectors-negative300.bin', binary=True)

vocab = model.vocab.keys()
wordsInVocab = len(vocab)
print (wordsInVocab)


#%%

# Converts the gensim model to a numpy array.
# and corresponding labels.
# vectors = np.asarray(model.wv.vectors)
# labels = np.asarray(model.wv.index2word)

# Saves this (large) numpy array to file.
# with open('w2v_as_np.pickle', 'wb') as f:
#     pickle.dump(vectors, f)
#
# with open('labels_for_npmat.pickle', 'wb') as f:
#     pickle.dump(labels, f)

#%% Loads the saved numpy array and labels.
with open('w2v_as_np.pickle', 'rb') as f:
    vectors = pickle.load(f)

with open('labels_for_npmat.pickle', 'rb') as f:
    labels = pickle.load(f)


#%% Declares the gender subspace pairs, then finds the subspace and plots the sigmas.

gender_pair_list = [ ('she', 'he'), ('her', 'his'), ('woman', 'man'), ('Mary', 'John'), ('herself', 'himself'),
                     ('daughter', 'son'), ('mother', 'father'), ('gal', 'guy'), ('girl', 'boy'), ('female', 'male')]

gender_subspace = helpers.get_subspace(labels, vectors, gender_pair_list)
U, S, VT = helpers.norm_svd(gender_subspace)

#%%
helpers.sing_value_plot(S)
plt.show()

#%%

gender_direction = U[:,1]

#%%
word = 'he'
bias = helpers.get_direct_bias(word, gender_direction, labels, vectors)
print(bias)