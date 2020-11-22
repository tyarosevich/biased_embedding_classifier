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
from sklearn.model_selection import train_test_split

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

#%% Set up the training data.

# Get the 500 largest and smallest differences between the
# cleaned and uncleaned word2vec sets. Largest will be assumed
# to be 'biased', smallest 'unbiased'.
idx_biased = helpers.n_largest(500, cos_sim)
idx_unbiased = helpers.n_largest(500, cos_sim, smallest=True)

# Make lists/arrays of the labels as indices from the original
# word2vec corpus, the strings of the words, and the binary labels.
labels_corpus_ind = np.concatenate( (idx_unbiased, idx_biased) )
labels_strings = list(labels[idx_unbiased]) + list(labels[idx_biased])
labels_binary = np.concatenate( (np.zeros( (500,)), np.ones( (500,) )))

# Extract the word embeddings and concatenate into full training
# matrix.
unbiased_vecs = vectors_short[:, idx_unbiased]
biased_vecs = vectors_short[:, idx_biased]
full_training_mat = np.hstack( (unbiased_vecs, biased_vecs) )


#%% List analogies from the original (i.e. still biased) w2v corpus. A cursory
# examination shows that these least-changed embeddings are indeed not biased,
# since the analogies return sensible results such as 'he : cooking -> she : Cooking'.
for word in labels_strings[0:50]:
    print(helpers.he_she_compare(model, word))

#%% Does the same with the largest cosine similarity embeddings. The results
# from this one are far less clear, and cast doubt on the entire endeavor.
for word in labels_strings[-50:]:
    print(helpers.he_she_compare(model, word))

#%% Create train/test sets

trans_mat = full_training_mat.T
x_train, x_test, y_train, y_test = train_test_split(trans_mat, labels_binary, test_size=0.2, random_state=13)

#%% Imports for training

from keras.models import Sequential
from keras.layers import Dense
from keras import Input
from tensorflow.keras import initializers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

#%% Network architecture

def create_base_model():
    base_model = Sequential()
    base_model.add(Dense(20, activation = 'relu', kernel_initializer= 'random_normal', input_dim=300))
    base_model.add(Dense(20, activation = 'relu', kernel_initializer= 'random_normal'))
    base_model.add(Dense(1, activation='sigmoid',kernel_initializer= 'random_normal'))
    base_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return base_model

#%% Run model
classif_output = KerasClassifier(build_fn=create_base_model, epochs=100, batch_size=10, verbose=0, )

# K=10 fold cross validation
k_fold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(classif_output, x_train, y_train, cv=k_fold)
print("Accuracy: %.2f%%   STD: %.2f%%" % (results.mean()*100, results.std()*100))
#%% Also fit the model directly for method access.
classif_output.fit(x_train, y_train)

#%% Evaluate

y_pred = classif_output.model.predict(x_test)
y_pred = (y_pred > 0.5)
confmat = confusion_matrix(y_test, y_pred)
print(confmat)
