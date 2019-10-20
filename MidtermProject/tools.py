# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/19 22:56

import re
import string

from nltk.stem.wordnet import WordNetLemmatizer


def load_stopwords():
    with open('./stopwords_eng.txt', 'r') as f:
        word_set = []
        for line in f.readlines():
            word_set.append(line.strip())
        return frozenset(word_set)


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
