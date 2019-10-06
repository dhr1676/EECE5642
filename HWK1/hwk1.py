# _*_coding:utf-8_*_
# Author    : Ding
# Time      : 2019/9/25 21:51

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COL_NAME = [
    "name",
    "manufacturer",
    "type",
    "calories",
    "protein",
    "fat",
    "sodium",
    "dietary_fiber",
    "carbohydrates",
    "sugars",
    "display_shelf",
    "potassium",
    "vitamins_minerals",
    "weight",
    "cups"
]

raw_data = pd.read_csv("./cereal.csv",
                       sep=' ',
                       header=None,
                       names=COL_NAME,
                       engine="python")

print(raw_data.shape)

# ##############################################################################
# Nutrient Pie Chart
nutrient_labels = ["protein",
                   "fat",
                   "dietary_fiber",
                   "carbohydrates",
                   "sugars"
                   ]
nutrient_data = raw_data[nutrient_labels]
explode = [0, 0, 0, 0, 0.1]
for i in range(3):
    x = nutrient_data.iloc[i].tolist()
    plt.pie(x, labels=nutrient_labels, explode=explode)
    plt.title("Nutrient Pie Chart for " + raw_data.iloc[i]["name"], fontsize='large', fontweight='bold')
    plt.show()
    plt.close()
# ##############################################################################

# ##############################################################################
# Nutrient Histogram
sugar_data = raw_data[["name", "sugars", "protein"]].iloc[0:10]
x = range(len(sugar_data))
sugar_list = sugar_data.sugars.tolist()
protein_list = sugar_data.protein.tolist()

plt.bar(x, sugar_list)
plt.xticks(x, ["Brand" + str(idx) for idx in range(10)])
plt.tick_params(labelsize=8)
plt.title("Sugar Amount Per Serving", fontsize='large', fontweight='bold')
plt.show()
plt.close()

plt.bar(x, protein_list)
plt.xticks(x, ["Brand" + str(idx) for idx in range(10)])
plt.tick_params(labelsize=8)
plt.title("Protein Amount Per Serving", fontsize='large', fontweight='bold')
plt.show()
plt.close()

l1 = plt.bar(x, sugar_list, alpha=0.7, width=0.3, color='green')
l2 = plt.bar(x, protein_list, alpha=0.7, width=0.3, color='red', bottom=sugar_list)
plt.legend(handles=[l1, l2], labels=['Sugars', 'Protein'], loc='best')
plt.xticks(x, ["Brand" + str(idx) for idx in range(10)])
plt.tick_params(labelsize=8)
plt.title("Sugar and Protein Amount Per Serving", fontsize='large', fontweight='bold')
plt.show()
plt.close()

plt.bar(np.arange(10), sugar_list, alpha=0.7, width=0.3, color='green', label='Sugars', lw=3)
plt.bar(np.arange(10) + 0.4, protein_list, alpha=0.7, width=0.3, color='red', label='Protein', lw=3)
plt.legend(loc='best')
plt.xticks(x, ["Brand" + str(idx) for idx in range(10)])
plt.tick_params(labelsize=8)
plt.title("Sugar and Protein Amount Per Serving", fontsize='large', fontweight='bold')
plt.show()
plt.close()

# ##############################################################################
