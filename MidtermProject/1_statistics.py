# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/10/5 13:51


from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import matplotlib.pyplot as plt

# news_train = fetch_20newsgroups(subset='train')
# print("Number of articles: " + str(len(news_train.data)))
# print("Number of different categories: " + str(len(news_train.target_names)))
# # Number of articles: 11314
# # Number of different categories: 20
# pprint(list(news_train.target_names))

target_names = ['alt.atheism',
                'comp.graphics',
                'comp.os.ms-windows.misc',
                'comp.sys.ibm.pc.hardware',
                'comp.sys.mac.hardware',
                'comp.windows.x',
                'misc.forsale',
                'rec.autos',
                'rec.motorcycles',
                'rec.sport.baseball',
                'rec.sport.hockey',
                'sci.crypt',
                'sci.electronics',
                'sci.med',
                'sci.space',
                'soc.religion.christian',
                'talk.politics.guns',
                'talk.politics.mideast',
                'talk.politics.misc',
                'talk.religion.misc']

# pie_chart = []
# for i in range(len(target_names)):
#     parts = [target_names[i]]
#     part = fetch_20newsgroups(subset='train', categories=parts)
#     # print(parts, part.filenames.shape)
#     pie_chart.append(len(part.filenames))

# ['alt.atheism'] (480,)
# ['comp.graphics'] (584,)
# ['comp.os.ms-windows.misc'] (591,)
# ['comp.sys.ibm.pc.hardware'] (590,)
# ['comp.sys.mac.hardware'] (578,)
# ['comp.windows.x'] (593,)
# ['misc.forsale'] (585,)
# ['rec.autos'] (594,)
# ['rec.motorcycles'] (598,)
# ['rec.sport.baseball'] (597,)
# ['rec.sport.hockey'] (600,)
# ['sci.crypt'] (595,)
# ['sci.electronics'] (591,)
# ['sci.med'] (594,)
# ['sci.space'] (593,)
# ['soc.religion.christian'] (599,)
# ['talk.politics.guns'] (546,)
# ['talk.politics.mideast'] (564,)
# ['talk.politics.misc'] (465,)
# ['talk.religion.misc'] (377,)

# x = pie_chart
# plt.pie(x, labels=target_names, autopct='%.1f%%', startangle=90, counterclock=False)
# plt.title("Pie Chart for 20 Newsgroup", fontsize='large', fontweight='bold')
# plt.show()
# plt.close()


parts = ['alt.atheism']  # 480
part = fetch_20newsgroups(subset='train', categories=parts)
# part.data is a list
pprint(part.data[0])
