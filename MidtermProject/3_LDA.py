# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/7 13:37

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


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    print("#主题-词语分布矩阵: \n", model.components_)


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

    # #### Train LDA models for BoW
    num_topics = 20
    lda_bow = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
    lda_bow.fit(data_bow)
    display_topics(lda_bow, feature_names_bow, num_topics)

    print('\n\n\n')

    # #### Train LDA models for TF-IDF
    num_topics = 20
    lda_tfidf = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online',
                                          learning_offset=50.,
                                          random_state=0)
    lda_tfidf.fit(data_tfidf)
    display_topics(lda_tfidf, feature_names_tfidf, num_topics)


if __name__ == '__main__':
    main()
