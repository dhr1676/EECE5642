# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/7 13:37

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Doc2Vec
from pprint import pprint
from MidtermProject.tools import load_stopwords, pre_processing
import gensim


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

    # for i in range(5):
    #     print(processed_data[i])
    #     print("\n\n")
    # print("\n\n\n\n\n")

    max_epochs = 500
    vec_size = 20
    alpha = 0.025

    model = gensim.models.Doc2Vec(processed_data, dm=0, alpha=0.1, size=20, min_alpha=0.025)

    # d2v_model = Doc2Vec(vector_size=vec_size, min_count=1, dm=1, alpha=alpha, min_alpha=0.00025)
    #
    # d2v_model.build_vocab(processed_data)


if __name__ == '__main__':
    main()
