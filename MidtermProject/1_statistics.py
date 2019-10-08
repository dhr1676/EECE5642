# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/5 13:51


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from pprint import pprint
import re
import string
from MidtermProject.stopwords import load_stopwords


def pre_processing(text):
    # #### Remove punctuations 去除标点
    text = re.sub(r'[{}]+'.format(string.punctuation), ' ', text)
    text = text.strip().lower()

    # #### Remove numbers 去除数字
    # remove_digits = str.maketrans('', '', string.digits)
    # text = text.translate(remove_digits)
    text = re.sub(r'[{}]+'.format(string.digits), ' ', text)

    # #### Lemmatize 把英语词汇归元化/标准化
    lemma = WordNetLemmatizer()
    text = " ".join([lemma.lemmatize(word) for word in text.split()])

    return text


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

    processed_data = [pre_processing(data) for data in train_data]
    stopwords = load_stopwords()

    # #### Learn Bag-of-words (BoW)
    count_vec = CountVectorizer(strip_accents='unicode',
                                stop_words=stopwords)
    count_vec.fit(processed_data)
    data_bow = count_vec.transform(processed_data)

    feature_names_bow = count_vec.get_feature_names()
    print(len(processed_data), data_bow.shape, type(data_bow))
    pprint(count_vec.vocabulary_)

    print('对应特征总个数：', (data_bow.toarray().sum(axis=0)))
    key_words = {}
    # for idx, w_idx in enumerate(data_bow.toarray().sum(axis=0)):
    #     key_words


if __name__ == '__main__':
    main()
