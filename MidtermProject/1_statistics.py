# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/5 13:51


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from pprint import pprint
import re
import string
import matplotlib.pyplot as plt
from MidtermProject.tools import load_stopwords, pre_processing


def main():
    # news_train = fetch_20newsgroups(subset='train')
    # print("Number of articles: " + str(len(news_train.data)))
    # print("Number of different categories: " + str(len(news_train.target_names)))
    # # Number of articles: 11314
    # # Number of different categories: 20
    # pprint(list(news_train.target_names))

    # target_names = ['alt.atheism',
    #                 'comp.graphics',
    #                 'comp.os.ms-windows.misc',
    #                 'comp.sys.ibm.pc.hardware',
    #                 'comp.sys.mac.hardware',
    #                 'comp.windows.x',
    #                 'misc.forsale',
    #                 'rec.autos',
    #                 'rec.motorcycles',
    #                 'rec.sport.baseball',
    #                 'rec.sport.hockey',
    #                 'sci.crypt',
    #                 'sci.electronics',
    #                 'sci.med',
    #                 'sci.space',
    #                 'soc.religion.christian',
    #                 'talk.politics.guns',
    #                 'talk.politics.mideast',
    #                 'talk.politics.misc',
    #                 'talk.religion.misc']

    # pie_chart = []
    # for i in range(len(target_names)):
    #     parts = [target_names[i]]
    #     part = fetch_20newsgroups(subset='train', categories=parts)
    #     # print(parts, part.filenames.shape)
    #     pie_chart.append(len(part.filenames))

    # x = pie_chart
    # plt.pie(x, labels=target_names, autopct='%.1f%%', startangle=90, counterclock=False)
    # plt.title("Pie Chart for 20 Newsgroup", fontsize='large', fontweight='bold')
    # plt.show()
    # plt.close()

    # parts = ['alt.atheism']  # 480
    # part = fetch_20newsgroups(subset='train', categories=parts)
    # # part.data is a list
    # raw = part.data[0]
    # print(raw, "--------------------------------------------------------\n\n\n\n")
    #
    # processed = pre_processing(raw)
    # print(processed)

    news_train = fetch_20newsgroups(subset='train')
    train_data = news_train.data
    print(len(train_data))
    print(type(train_data), type(train_data[0]), "\n")  # <class 'list'> <class 'str'>

    processed_data = [" ".join(pre_processing(data)) for data in train_data]
    stopwords = load_stopwords()

    # #### Learn Bag-of-words (BoW)
    max_df = 0.8
    min_df = 0.001
    count_vec = CountVectorizer(strip_accents='unicode',
                                stop_words=stopwords,
                                max_df=max_df,
                                min_df=min_df)
    count_vec.fit(processed_data)
    data_bow = count_vec.transform(processed_data)

    feature_names_bow = count_vec.get_feature_names()
    print(len(processed_data), data_bow.shape, type(data_bow))

    freq_list = data_bow.toarray().sum(axis=0).tolist()

    print('对应特征总个数：', len(freq_list), type(freq_list))
    # 对应特征总个数： 90517 <class 'numpy.ndarray'>
    # array([  3, 239,  75,   2,   1,   1,   1,   2,   2,   1], dtype=int64)

    words = count_vec.vocabulary_
    index2words = dict(zip(words.values(), words.keys()))

    freq_dict = {}
    top_dict = {}
    for idx, freq in enumerate(freq_list):
        top_dict[index2words[idx]] = freq

    k = 20
    all = len(freq_list)
    top_k_list = sorted(top_dict.items(), key=lambda l: l[1], reverse=True)[:k]

    print(top_k_list)

    x = range(len(top_k_list))
    x_list = [t[1] for t in top_k_list]
    y_list = [t[0] for t in top_k_list]

    plt.bar(x, x_list)
    plt.xticks(x, y_list, rotation=45)
    plt.tick_params(labelsize=8)
    plt.title("Top %d / %d Words of  with max_df = %.2f and min_df = %.3f" % (k, all, max_df, min_df),
              fontsize='large', fontweight='bold')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
