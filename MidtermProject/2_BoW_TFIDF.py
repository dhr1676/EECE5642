# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/6 0:20

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn
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

    # for i in range(5):
    #     print(processed_data[i])
    #     print("\n\n")
    # print("\n\n\n\n\n")

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


if __name__ == '__main__':
    main()
