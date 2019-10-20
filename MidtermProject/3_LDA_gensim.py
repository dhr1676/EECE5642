# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/19 21:26

# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/7 13:37

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn
from nltk.stem.wordnet import WordNetLemmatizer
import wordcloud

from gensim import corpora
from gensim.models.ldamodel import LdaModel
import gensim

import re
import string
import time
from random import randint
from pprint import pprint

from MidtermProject.stopwords import load_stopwords

FEATURE_NUM = 5000


def pre_processing(text):
    # #### Remove punctuations 去除标点
    text = text.replace("\'", "")
    text = re.sub(r'[{}]+'.format(string.punctuation), ' ', text)
    text = text.strip().lower()

    # #### Remove numbers 去除数字
    # remove_digits = str.maketrans('', '', string.digits)
    # text = text.translate(remove_digits)
    text = re.sub(r'[{}]+'.format(string.digits), ' ', text)

    stop = load_stopwords()
    # text = " ".join([i for i in text.split() if i not in stop])
    text = [i for i in text.split() if i not in stop]

    # # #### Lemmatize 把英语词汇归元化/标准化
    # lemma = WordNetLemmatizer()
    # text = " ".join([lemma.lemmatize(word) for word in text.split()])

    return text


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
    remove = ('headers', 'footers', 'quotes')
    news_train = fetch_20newsgroups(subset='train', categories=target_names, remove=remove)
    train_data = news_train.data
    print(len(train_data))
    print(type(train_data), type(train_data[0]), "\n")  # <class 'list'> <class 'str'>

    processed_data = [pre_processing(data) for data in train_data]
    stopwords = load_stopwords()

    dictionary = corpora.Dictionary(processed_data)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_data]

    # LdaModel = gensim.models.ldamodel.LdaModel
    lda_model = LdaModel(doc_term_matrix, num_topics=5, id2word=dictionary, passes=50)

    # print(lda_model.print_topics(num_topics=5, num_words=10))

    for topic in lda_model.print_topics(num_words=10):
        print(topic)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Time cost: %.2fs" % (time.time() - start_time))
