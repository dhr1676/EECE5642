# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/7 13:37


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.manifold import TSNE
from nltk.stem.wordnet import WordNetLemmatizer

from pprint import pprint
import pandas as pd
import re
import string
import matplotlib.pyplot as plt

from MidtermProject.tools import load_stopwords, pre_processing


def main():
    target_names = ['alt.atheism',
                    'comp.graphics',
                    'rec.autos',
                    'sci.med',
                    'talk.politics.guns'
                    ]
    news_train = fetch_20newsgroups(subset='train',
                                    categories=target_names,
                                    remove=('headers', 'footers', 'quotes'))
    # news_train = fetch_20newsgroups(subset='train')
    train_data = news_train.data
    print(len(train_data))
    print(type(train_data), type(train_data[0]), "\n")  # <class 'list'> <class 'str'>

    processed_data = [pre_processing(data) for data in train_data]
    stopwords = load_stopwords()

    # # #######################################################################
    # #### Learn Bag-of-words (BoW)
    count_vec = CountVectorizer(stop_words=stopwords)
    count_vec.fit(processed_data)
    data_bow = count_vec.transform(processed_data)
    feature_names_bow = count_vec.get_feature_names()
    print(len(processed_data), data_bow.shape, type(data_bow))
    # pprint(count_vec.get_feature_names())
    # pprint(count_vec.vocabulary_)

    print('Start K-means for BoW:')
    num_clusters = 5
    weight_bow = data_bow.toarray()
    clf_bow = KMeans(n_clusters=num_clusters)
    s = clf_bow.fit(weight_bow)
    print(s)
    # 5个中心点
    print(clf_bow.cluster_centers_)
    # Sum of distances of samples to their closest cluster center
    print("Inertia of BoW", clf_bow.inertia_)

    # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(weight_bow)
    x, y = [], []
    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    plt.scatter(x, y, c=clf_bow.labels_, marker="x")
    plt.xticks(())
    plt.yticks(())
    plt.show()
    plt.savefig('./Chart/kMeans_sample_BoW.png', aspect=1)


if __name__ == '__main__':
    main()
