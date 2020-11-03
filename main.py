#%%

import gensim

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

# Returns a list of tuples of w1 + w2 - w3 and their nearest words.
def x_y_minusz(w1, w2, w3, n):
    result = model.most_similar(positive = [w1, w2], negative = [w3], topn = n)
    return result

#%%

x_y_minusz('woman', 'king', 'man', 1)


# Displays the cosine similarity between two words.
def compare_words(word1, word2):
    result = model.similarity(word1, word2)
    return result

# Returns a list of tuples of w1 + w2 - w3 and their nearest words.
def x_y_minusz(w1, w2, w3, n):
    result = model.most_similar(positive = [w1, w2], negative = [w3])