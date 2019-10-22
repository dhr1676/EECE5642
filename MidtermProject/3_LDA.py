# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/7 13:37

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn
from nltk.stem.wordnet import WordNetLemmatizer
import wordcloud

import re
import string
import time
from random import randint
from pprint import pprint

from MidtermProject.tools import load_stopwords, pre_processing

FEATURE_NUM = 5000


def display_topics(model, feature_names, no_top_words, file_name):
    topic_words = None
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % topic_idx)
        topic_words = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        print(topic_words)

        word_cloud = wordcloud.WordCloud(background_color='black',
                                         # color_func=random_color_func
                                         )
        word_cloud.generate(topic_words)
        word_cloud.to_file("./Chart/" + file_name + "_" + str(FEATURE_NUM) + "_" + str(topic_idx) + ".png")

    print("\n主题-词语分布矩阵:")
    for line in model.components_:
        print(type(line), len(line))


def random_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None,
                      random_state=None):
    h = randint(120, 250)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(randint(60, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)


def main():
    target_names = ['alt.atheism',
                    'comp.graphics',
                    'rec.autos',
                    'sci.med',
                    'talk.politics.guns'
                    ]
    news_train = fetch_20newsgroups(subset='train', categories=target_names)
    train_data = news_train.data
    print(len(train_data))
    print(type(train_data), type(train_data[0]), "\n")  # <class 'list'> <class 'str'>

    processed_data = [" ".join(pre_processing(data)) for data in train_data]
    stopwords = load_stopwords()

    max_df = 0.3
    min_df = 0.005

    # #### Learn Bag-of-words (BoW)
    # count_vec = CountVectorizer(stop_words='english')
    count_vec = CountVectorizer(stop_words=stopwords,
                                max_df=max_df,
                                min_df=min_df,
                                max_features=FEATURE_NUM)
    count_vec.fit(processed_data)
    data_bow = count_vec.transform(processed_data)
    feature_names_bow = count_vec.get_feature_names()
    print(len(processed_data), data_bow.shape, type(data_bow))

    # #### Learn TF-IDF model
    # tfidf_vec = TfidfVectorizer(stop_words='english')
    tfidf_vec = TfidfVectorizer(stop_words=stopwords,
                                max_df=max_df,
                                min_df=min_df,
                                max_features=FEATURE_NUM)
    tfidf_vec.fit(processed_data)
    data_tfidf = tfidf_vec.transform(processed_data)
    feature_names_tfidf = tfidf_vec.get_feature_names()
    print(len(processed_data), data_tfidf.shape, type(data_tfidf))

    # #### Train LDA models for BoW
    num_topics = 5
    num_topic_word = 20
    lda_bow = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
    lda_bow.fit(data_bow)
    display_topics(lda_bow, feature_names_bow, num_topic_word, "BoW_word_cloud")

    print('\n\n\n')

    # #### Train LDA models for TF-IDF
    num_topics = 5
    num_topic_word = 20
    lda_tfidf = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online',
                                          learning_offset=50.,
                                          random_state=0)
    lda_tfidf.fit(data_tfidf)
    display_topics(lda_tfidf, feature_names_tfidf, num_topic_word, "TF-IDF_word_cloud")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Time cost: %.2fs" % (time.time() - start_time))
