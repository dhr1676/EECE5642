# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/7 13:37


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
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

    # #######################################################################
    # #### Learn TF-IDF model
    tfidf_vec = TfidfVectorizer(stop_words=stopwords)
    tfidf_vec.fit(processed_data)
    data_tfidf = tfidf_vec.transform(processed_data)
    feature_names_tfidf = tfidf_vec.get_feature_names()
    print(len(processed_data), data_tfidf.shape, type(data_tfidf))

    print('Start K-means for TF-IDF:')
    weight_tfidf = data_tfidf.toarray()
    clf_tfidf = KMeans(n_clusters=5)
    s = clf_tfidf.fit(weight_tfidf)
    print(s)
    # 5个中心点
    print(clf_tfidf.cluster_centers_)
    # # 每个样本所属的簇
    # print(clf.labels_)
    # i = 1
    # while i <= len(clf.labels_):
    #     print(i, clf.labels_[i - 1])
    #     i = i + 1
    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    # Sum of distances of samples to their closest cluster center
    print("Inertia of TF-IDF", clf_tfidf.inertia_)

    # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(weight_tfidf)
    x, y = [], []
    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    plt.scatter(x, y, c=clf_tfidf.labels_, marker="x")
    plt.xticks(())
    plt.yticks(())
    plt.show()
    plt.savefig('./Chart/kMeans_sample_TFIDF.png', aspect=1)


if __name__ == '__main__':
    main()
