[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
<img src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" />
<img src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/>

# Classifying word2vec embeddings as 'biased' or 'unbiased
This project is a small, highly informal experiment examining the results of the 2016 paper "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings." <sup>[1](#myfootnote1)</sup> The intention is to use the cosine-similarity of the original word2vec embeddings against the authors' debiased embedding set to boostrap enough labeled data to make use of a deep neural net classifier. 

# Motivation
The possibility that the linear transformation used by the authors of the original paper (which is elegant, efficient, and brilliant) might not remove bias from the set of all biased words, since it seems unlikely to me that a subspace derived from the PCA of a limited number of embeddings could truly account for the semantically varied forms that gender bias takes in language.

# Framework
Python - 3.8
numpy - 1.19.2 
Scikit-Learn - 0.23.2
keras - 2.4.3

# Code Example

```
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
```



<a name="myfootnote1">1</a>:Bolukbasi, Tolga, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama and A. Kalai. “Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings.” ArXiv abs/1607.06520 (2016): n. pag.