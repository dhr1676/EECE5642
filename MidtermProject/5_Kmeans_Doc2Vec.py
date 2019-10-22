# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/7 13:37


from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import numpy as np
import matplotlib.pyplot as plt

from MidtermProject.tools import *


def clean_text(texts, min_length=2):
    clean = []
    # don't remove apostrophes
    translator = str.maketrans(string.punctuation.replace('\'', ' '), ' ' * len(string.punctuation))
    for text in texts:
        text = text.translate(translator)
        tokens = text.split()
        # remove not alphabetic tokens
        tokens = [word.lower() for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = stopwords.words('english')
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) >= min_length]
        tokens = ' '.join(tokens)
        clean.append(tokens)
    return clean


def tag_text(all_text, tag_type=''):
    tagged_text = []
    tag_list = []
    for i, text in enumerate(all_text):
        # tag = tag_type + '_' + str(i)
        tag_list.append(i)
        tag = str(i)
        tagged_text.append(TaggedDocument(text.split(), [tag]))
    return tagged_text, tag_list


def train_docvec(dm, dbow_words, min_count, epochs, training_data):
    vec_size = 100
    model = Doc2Vec(dm=dm, dbow_words=dbow_words, min_count=min_count)
    model.build_vocab(training_data)
    model.train(training_data, total_examples=len(training_data), epochs=epochs)
    return model


def main():
    target_names = ['alt.atheism',
                    'comp.graphics',
                    'rec.autos',
                    'sci.med',
                    'talk.politics.guns'
                    ]
    dataset = fetch_20newsgroups(subset='train',
                                 categories=target_names,
                                 remove=('headers', 'footers', 'quotes'))
    n_samples = len(dataset.data)
    data = clean_text(dataset.data)
    tagged_data, tagged_list = tag_text(data)
    data_labels = dataset.target
    data_label_names = dataset.target_names

    model_DBOW = train_docvec(0, 0, 4, 30, tagged_data)

    X_train = np.array([model_DBOW.docvecs[str(i)] for i in range(len(tagged_data))])

    # # #######################################################################
    print('Start K-means for Doc2Vec:')
    num_clusters = 5
    weight_doc2vec = X_train
    clf_doc2vec = KMeans(n_clusters=num_clusters)
    s = clf_doc2vec.fit(weight_doc2vec)
    print(s)
    # 5个中心点
    print(clf_doc2vec.cluster_centers_)
    # Sum of distances of samples to their closest cluster center
    print("Inertia of Doc2Vec", clf_doc2vec.inertia_)

    # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(weight_doc2vec)
    x, y = [], []
    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    plt.scatter(x, y, c=clf_doc2vec.labels_, marker="x")
    plt.xticks(())
    plt.yticks(())
    plt.show()
    plt.savefig('./Chart/kMeans_sample_Doc2Vec.png', aspect=1)


if __name__ == '__main__':
    main()
