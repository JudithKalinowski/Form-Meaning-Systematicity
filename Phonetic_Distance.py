"""

PHONETIC DISTANCE

@author: Judith Kalinowski

"""

import numpy as np
import pandas as pd
from Levenshtein import distance as lev

items_df = pd.read_csv("./word_df.csv", header=int())
save_to_csv = './csv/'

''' in the following, we work on the IPA transcriptions of the IPA dataframe
so that we can use it for the levenshtein distance '''

items_lst_IPA = pd.DataFrame(items_df, columns=['num_item_id', 'IPA'])
items_lst_IPA.index = np.arange(0, len(items_lst_IPA))
for row in range(len(items_lst_IPA)):
    'remove all numbers'
    if items_lst_IPA.at[row, "IPA"] in ["1", "2", "3"]:
        items_lst_IPA.at[row, "IPA"] = items_df.at[row, "definition_new"]
    'remove all round brackets'
    bracket = "()[]"
    for char in bracket:
        items_lst_IPA.at[row, "IPA"] = items_lst_IPA.at[row, "IPA"].replace(char, "")
    'remove all but the first word options indicated by "/"'
    sep = "/"
    items_lst_IPA.at[row, "IPA"] = items_lst_IPA.at[row, "IPA"].split(sep, 1)[0]
IPA_lst = items_lst_IPA.IPA.tolist()


lev_dist_df = pd.DataFrame()
for word1 in IPA_lst:
    dist_word1 = []
    for word2 in IPA_lst:
        dist_1_2 = lev(word1, word2)
        dist_word1.append(dist_1_2)
    lev_dist_df[word1] = dist_word1

lev_dist_df.to_csv(save_to_csv + 'pairwise_lev_dist_df.csv')