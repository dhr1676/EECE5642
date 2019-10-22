# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/19 23:01

from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
from gensim import models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
import numpy as np
from nltk.corpus import stopwords
from string import punctuation

from pprint import pprint


def clean_text(texts, min_length=2):
    clean = []
    # don't remove apostrophes
    translator = str.maketrans(punctuation.replace('\'', ' '), ' ' * len(punctuation))
    for text in texts:
        text = text.translate(translator)
        tokens = text.split()
        # remove not alphabetic tokens
        tokens = [word.lower() for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = stopwords.words('english')
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) >= min_length]
        tokens = ' '.join(tokens)
        clean.append(tokens)
    return clean


def tag_text(all_text, tag_type=''):
    tagged_text = []
    tag_list = []
    for i, text in enumerate(all_text):
        # tag = tag_type + '_' + str(i)
        tag_list.append(i)
        tag = str(i)
        tagged_text.append(models.doc2vec.TaggedDocument(text.split(), [tag]))
    return tagged_text, tag_list


def train_docvec(dm, dbow_words, min_count, epochs, training_data):
    vec_size = 100
    model = models.Doc2Vec(dm=dm, dbow_words=dbow_words, min_count=min_count)
    model.build_vocab(tagged_data)
    model.train(training_data, total_examples=len(training_data), epochs=epochs)
    return model


def compare_vectors(vector1, vector2):
    cos_distances = []
    for i in range(len(vector1)):
        d = distance.cosine(vector1[i], vector2[i])
        cos_distances.append(d)
    print(np.median(cos_distances))
    print(np.std(cos_distances))


dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
n_samples = len(dataset.data)
data = clean_text(dataset.data)
tagged_data, tagged_list = tag_text(data)
data_labels = dataset.target
data_label_names = dataset.target_names

model_DBOW = train_docvec(0, 0, 4, 30, tagged_data)

pprint(len(tagged_data))
pprint(tagged_data[0:4])
# 获取与tag为0的向量最相似的数据
sims = model_DBOW.docvecs.most_similar(0)
pprint(sims)
# 计算tag 0，2两者之间的距离
pprint(model_DBOW.docvecs.similarity(0, 2))
# 获取tag 4的向量
pprint(model_DBOW.docvecs[4])

# # #### #############################################################
# X_train = np.array([model_DBOW.docvecs[str(i)] for i in range(len(tagged_data))])
# y_train = list(data_label_names)
#
# print(X_train.shape)
# print(len(y_train))
#
# # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
# tsne = TSNE(n_components=2)
# decomposition_data = tsne.fit_transform(X_train)
# x, y = [], []
# for i in decomposition_data:
#     x.append(i[0])
#     y.append(i[1])
# fig = plt.figure(figsize=(10, 10))
# ax = plt.axes()
# plt.scatter(x, y, c=tagged_list, marker="x")
# plt.xticks(())
# plt.yticks(())
# plt.show()

# model_dbow2 = train_docvec(0, 0, 4, 30, tagged_data)
# model_dbow3 = train_docvec(0, 1, 4, 30, tagged_data)
# model_dbow4 = train_docvec(0, 1, 4, 30, tagged_data)
# model_dm1 = train_docvec(1, 0, 4, 30, tagged_data)
# model_dm2 = train_docvec(1, 0, 4, 30, tagged_data)

# compare_vectors(model_dbow1.docvecs, model_dbow2.docvecs)
