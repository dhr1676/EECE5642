# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/7 13:37


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem.wordnet import WordNetLemmatizer

from pprint import pprint

import re
import string


def pre_processing(text):
    # #### Remove punctuations 去除标点
    text = re.sub(r'[{}]+'.format(string.punctuation), ' ', text)
    text = text.strip().lower()

    # #### Remove numbers 去除数字
    remove_digits = str.maketrans('', '', string.digits)
    text = text.translate(remove_digits)

    # #### Lemmatize 把英语词汇归元化/标准化
    lemma = WordNetLemmatizer()
    normalized = " ".join([lemma.lemmatize(word) for word in text.split()])

    return normalized


def main():
    target_names = ['alt.atheism']
    news_train = fetch_20newsgroups(subset='train', categories=target_names)
    train_data = news_train.data
    print(len(train_data))
    print(type(train_data), type(train_data[0]), "\n")  # <class 'list'> <class 'str'>

    processed_data = [pre_processing(data) for data in train_data]

    # #### Learn Bag-of-words (BoW)
    count_vec = CountVectorizer(stop_words='english')
    count_vec.fit(processed_data)
    data_bow = count_vec.transform(processed_data)
    feature_names_bow = count_vec.get_feature_names()
    print(len(processed_data), data_bow.shape, type(data_bow))

    # #### Learn TF-IDF model
    tfidf_vec = TfidfVectorizer(stop_words='english')
    tfidf_vec.fit(processed_data)
    data_tfidf = tfidf_vec.transform(processed_data)
    feature_names_tfidf = tfidf_vec.get_feature_names()
    print(len(processed_data), data_tfidf.shape, type(data_tfidf))

    # pprint(count_vec.get_feature_names())
    # pprint(count_vec.vocabulary_)

    print('Start K-means:')
    weight_tfidf = data_tfidf.toarray()

    clf = KMeans(n_clusters=20)
    s = clf.fit(weight_tfidf)
    print(s)

    # 20个中心点
    print(clf.cluster_centers_)

    # 每个样本所属的簇
    print(clf.labels_)
    i = 1
    while i <= len(clf.labels_):
        print(i, clf.labels_[i - 1])
        i = i + 1
        # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    print(clf.inertia_)


if __name__ == '__main__':
    main()
