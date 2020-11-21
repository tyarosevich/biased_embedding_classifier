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
from sklearn.metrics.pairwise import cosine_similarity

#%% Loads the word2vec news corpus into gensim to make use of gensims
# analytical tools.

model = gensim.models.KeyedVectors.load_word2vec_format('/home/tyarosevich/code_work/word2vec_news/GoogleNews-vectors-negative300.bin', binary=True)
#%%
# Load the debiased version
model_debiased = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-hard-debiased.bin', binary=True)

#%% Load the full embeddings corpus as a ndarray
vectors = helpers.load_stuff('w2v_as_np.pickle')
#%% load the full corpus labels as ndarray of strings
labels = helpers.load_stuff('labels_for_npmat.pickle')

#%% Load the full embeddings corpus debiased. Downloaded from
# https://github.com/tolga-b/debiaswe
vectors_debiased = helpers.load_stuff('w2v_as_np_debiased.pickle')

#%% Load the first 10,000 embeddings, both biased and debiased.
vectors_short_debiased = helpers.load_stuff('w2v_as_np_debiased_short.pickle')
vectors_short = helpers.load_stuff('w2v_as_np_short.pickle')

#%% Load the matrix of the difference of the norm (columnwise) of
# the biased and unbiased embeddings.
diff_mat = helpers.load_stuff('diff_bias_unbias')
#%% Loads the first 10,000 features of the unbiased
# corpus minus the biased corpus.
with open('unb_minus_bias_mat.pickle', 'rb') as f:
    diff_mat = pickle.load(f)
#%% Load JSON files from paper at above github
defs = helpers.load_stuff('debiaswe/data/definitional_pairs.json')
equalize_pairs = helpers.load_stuff('debiaswe/data/equalize_pairs.json')
gender_specific_words = helpers.load_stuff('debiaswe/data/gender_specific_full.json')
professions_list - helpers.load_stuff('debiaswe/data/professions.json')
#%% Declares the gender subspace pairs, then finds the subspace and plots the sigmas.

gender_pair_list = [ ('she', 'he'), ('her', 'his'), ('woman', 'man'), ('Mary', 'John'), ('herself', 'himself'),
                     ('daughter', 'son'), ('mother', 'father'), ('gal', 'guy'), ('girl', 'boy'), ('female', 'male')]

gender_subspace = helpers.get_subspace(labels, vectors, gender_pair_list)
U, S, VT = helpers.norm_svd(gender_subspace)

#%%
helpers.sing_value_plot(S)
plt.show()

#%%
gender_direction = helpers.get_vector('she', labels, vectors) - helpers.get_vector('he', labels, vectors)
gender_direction /= np.linalg.norm(gender_direction)
#%% To observe the difference between the two, we take the cosine similarity.
# Couldn't find a clearly documented and efficient method in any of the
# modern libraries, but the loop is very fast.
cos_sim = np.empty( (10000,))
for i in range(10000):
    cos_sim[i] = (vectors_short[:,i] @ vectors_short_debiased[:,i]) / (np.linalg.norm(vectors_short[:,i]) * np.linalg.norm(vectors_short_debiased[:,i]))


#%% Get indices of the n largest values and save list of
# corresponding words.
ind_test = helpers.n_largest(100, cos_sim)
ind_test_smallest = helpers.n_largest(100, cos_sim, smallest = True)

sample = labels[ind]
sample_smallest = labels[ind_smallest]