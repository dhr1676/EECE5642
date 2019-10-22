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
from gensim.models import Word2Vec
from gensim.models.ldamodel import LdaModel
import gensim

import re
import string
import time
from random import randint
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim
from matplotlib import pyplot

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
    remove = ('headers', 'footers', 'quotes')
    news_train = fetch_20newsgroups(subset='train', categories=target_names, remove=remove)
    train_data = news_train.data
    print(len(train_data))
    print(type(train_data), type(train_data[0]), "\n")  # <class 'list'> <class 'str'>

    processed_data = [pre_processing(data) for data in train_data]

    # Create Dictionary
    id2word = corpora.Dictionary(processed_data)

    # Create Corpus
    texts = processed_data

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=5,
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)

    pprint(lda_model.print_topics())
    # Visualize the topics
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, 'lda.html')

    # doc_lda = lda_model[corpus]

    # # train model
    # model = Word2Vec(processed_data[:10], iter=3)
    # # fit a 2d PCA model to the vectors
    # X = model[model.wv.vocab]
    # pca = PCA(n_components=2)
    # result = pca.fit_transform(X)
    # # create a scatter plot of the projection
    # pyplot.scatter(result[:, 0], result[:, 1])
    # words = list(model.wv.vocab)
    # for i, word in enumerate(words):
    #     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    # pyplot.show()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Time cost: %.2fs" % (time.time() - start_time))
