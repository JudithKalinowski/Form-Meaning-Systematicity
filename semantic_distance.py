# -*- coding: utf-8 -*-
"""

SEMANTIC DISTANCE

@author: Judith Kalinowski

"""

import pandas as pd
import fasttext
import fasttext.util
from scipy.spatial import distance


""" load the word vectors """

ft = fasttext.load_model('./sem_dist/cc.no.300.bin')
fasttext.util.reduce_model(ft, 100)
ft.save_model("model_filename.csv")

""" store all words of the downloaded word vectors in a list """

words_lst = ft.words
items_df = pd.read_csv("./word_df.csv", header=int())
items_lst = items_df.definition_new.tolist()


""" store all words in a list which are also in the wordbank CDI """

word_lst_red = [x for x in words_lst if x in items_lst]
items_lst_red = [x for x in items_lst if x in word_lst_red]

# fasttext does not include all words from the wordbank CDI but 35 less
# which we have to omit from further analysis in the voc_pred_functions.py
# and voc_pred_execution.py files

""" compute the cosine similarities of the word vectors and store in df"""

cos_sim_df = pd.DataFrame({})

for word_i in items_lst_red:
    lst = []
    for word_j in items_lst_red:
        sem_dist = distance.cosine(ft.get_word_vector(word_i), ft.get_word_vector(word_j))
        lst.append(sem_dist)
    series = pd.Series(lst)
    cos_sim_df = pd.concat([cos_sim_df, series], axis=1)


cos_sim_df = pd.concat([cos_sim_df, pd.Series(items_lst_red).rename('word')], axis=1)
#items_lst_red.append('word')
cos_sim_df = cos_sim_df.set_index('word')
cos_sim_df.columns = items_lst_red

cos_sim_df.to_csv('./sem_dist/cos_sim_df.csv')

""" get word vectors and store them in a dataframe """

word_vector_df = pd.DataFrame({})

for word in items_lst_red:
    get_word = ft.get_word_vector(word)
    get_word_to_df = pd.DataFrame(data=get_word)
    get_word_to_df.columns = columns=[word]
    word_vector_df = pd.concat([word_vector_df, get_word_to_df], axis=1, join='outer')


word_vector_df.to_csv('./sem_dist/word_vector_df.csv')






