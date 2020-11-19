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
import json


# reload helper functions because ipython is lame.
#%% Loads the word2vec news corpus into gensim to make use of gensims
# analytical tools.

model = gensim.models.KeyedVectors.load_word2vec_format('/home/tyarosevich/code_work/word2vec_news/GoogleNews-vectors-negative300.bin', binary=True)
print(len(vocab))
#%%
# Load the debiased version
model_debiased = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-hard-debiased.bin', binary=True)
print(len(vocab))

#%% Convert to numpy arrays and pickle

#Converts the gensim model to a numpy array.
# and corresponding labels.
# vectors_debiased = np.asarray(model_debiased.wv.vectors)
# labels_debiased = np.asarray(model.wv.index2word)
#
# # Saves this (large) numpy array to file.
# with open('w2v_as_np_debiased.pickle', 'wb') as f:
#     pickle.dump(vectors, f)
#
# with open('labels_debiased.pickle', 'wb') as f:
#     pickle.dump(labels, f)

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

#%% The difference set
vectors_short = vectors[:,0:10000]
vec_short_norm = vectors_short / np.linalg.norm(vectors_short, axis=0, ord=2)

#%% Loads the saved numpy array and labels.
with open('w2v_as_np.pickle', 'rb') as f:
    vectors = pickle.load(f)

with open('labels_for_npmat.pickle', 'rb') as f:
    labels = pickle.load(f)

with open('w2v_as_np_debiased.pickle', 'rb') as f:
    vectors_debiased = pickle.load(f)

with open('labels_debiased.pickle.pickle', 'rb') as f:
    labels_debiased = pickle.load(f)


#%% Declares the gender subspace pairs, then finds the subspace and plots the sigmas.

gender_pair_list = [ ('she', 'he'), ('her', 'his'), ('woman', 'man'), ('Mary', 'John'), ('herself', 'himself'),
                     ('daughter', 'son'), ('mother', 'father'), ('gal', 'guy'), ('girl', 'boy'), ('female', 'male')]

gender_subspace = helpers.get_subspace(labels, vectors, gender_pair_list)
U, S, VT = helpers.norm_svd(gender_subspace)

#%%
helpers.sing_value_plot(S)
plt.show()

#%%

# Load the JSON files

with open('debiaswe/data/definitional_pairs.json', "r") as f:
    defs = json.load(f)
with open('debiaswe/data/equalize_pairs.json', "r") as f:
    equalize_pairs = json.load(f)
with open('debiaswe/data/gender_specific_full.json', "r") as f:
    gender_specific_words = json.load(f)
with open('debiaswe/data/professions.json', "r") as f:
    professions_list = json.load(f)

#%%
gender_direction = helpers.get_vector('she', labels, vectors) - helpers.get_vector('he', labels, vectors)
gender_direction /= np.linalg.norm(gender_direction)

#%%




#
