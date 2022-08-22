# -*- coding: utf-8 -*-
"""

Created on Mon Jul 18 11:09:00 2022

@author: Judith Kalinowski
shared second authors: Michaela Vystrcilová, Laura Pede
@email: judith.kalinowski@uni-goettingen.de
I thank Max Burg (University of Goettingen) for his help in this matter.

CORRELATION

In this code, we aim to find correlations between the phonological similarity
    of word pairs and their age of acquisition.
    We first calculate the Levenshtein distance of word pairs and plot a
    heatmap afterwards.
    We then sort the heatmap using different metrics to make the correlations
    better visible.

"""

""" Load needed packages """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for plotting
from Levenshtein import distance as lev  # to calculate levenshtein distance
from collections import Counter

""" import own functions from voc_pred_functions """
from voc_pred_functions import cluster_mtx, norm_conf_mtx, plot_conf_mtx, keys_of_value

""" Load needed files """
folder = "./OneDrive/Desktop/Vocab_Prediction/SORTED/"

NOR_X = pd.read_csv("./NOR_X.csv", header=int(), index_col=0, sep=";")
NOR_y = pd.read_csv("./NOR_y.csv", header=int(), index_col=0, sep=";")
items_df = pd.read_csv("./word_df.csv", header=int())
sem_dist_df = pd.read_csv("./sem_dist/cos_sim_df.csv", header=int(), index_col=0)
word_df = pd.read_csv("./word_df.csv", header=int(), index_col=0)
weights_full_df = pd.read_csv("./csv/weights_full.csv", header=int(), index_col=0)
acc_scores_full_df = pd.read_csv("./csv/accuracy_score_full.csv", header=int(), index_col=0)

save_to = './plots/heatmaps/'

""" PART 1: PHONOLOGICAL DISTANCE """

''' in the following, we work on the IPA transcriptions of the IPA dataframe
so that we can use it for the levenshtein distance '''

items_lst = items_df.num_item_id.tolist()
id_to_str = pd.DataFrame(items_df, columns=['num_item_id', 'Translation', 'IPA'])
id_to_str = id_to_str.set_index(['num_item_id'])

items_lst_IPA = pd.DataFrame(items_df, columns=['num_item_id', 'IPA'])
items_lst_IPA.index = np.arange(0, len(items_lst_IPA))
for row in range(len(items_lst_IPA)):
    # remove all numbers
    if items_lst_IPA.at[row, "IPA"] in ["1", "2", "3"]:
        items_lst_IPA.at[row, "IPA"] = items_df.at[row, "definition_new"]
    # remove all round brackets
    bracket = "()[]"
    for char in bracket:
        items_lst_IPA.at[row, "IPA"] = items_lst_IPA.at[row, "IPA"].replace(char, "")
        # remove all but the first word options indicated by "/"
    sep = "/"
    items_lst_IPA.at[row, "IPA"] = items_lst_IPA.at[row, "IPA"].split(sep, 1)[0]
IPA_lst = items_lst_IPA.IPA.tolist()

""" calculate the levenshtein distance """
lev_dist_df = pd.DataFrame({})
n = 0

for word_1 in IPA_lst:
    lev_dist = []
    for word_2 in IPA_lst:
        lev_dist.append(lev(word_1, word_2))
    lev_dist_df[n] = lev_dist
    n += 1

lev_dist_df.columns = IPA_lst
# lev_dist_df.to_csv(save_to + 'pairwise_lev_dist_df.csv')

# plot heatmap
fig, ax = plt.subplots(figsize=(40, 30))
sns.heatmap(lev_dist_df, cmap="YlGnBu")
# ax.set_title('Pairwise Levenshtein distances', size=25)
# ax.tick_params(labelsize=20)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=50)

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# turn the axis label
for item in ax.get_yticklabels():
    item.set_rotation(0)
# save plot
fig.savefig(save_to + 'phon.png', bbox_inches='tight', dpi=200)

""" in order to work with the semantic distance matrix, we need to omit some
rows/cols, because the word embeddings were not available for all words (35 are missing)"""

sem_dist_word_lst = sem_dist_df.columns  # words in the sem dist matrix
CDI_word_lst = word_df.definition_new.to_list()  # words in the df we are using

lev_dist_df.columns = CDI_word_lst
lev_dist_df.index = CDI_word_lst
red_lev_dist_df = lev_dist_df.drop(columns=[col for col in lev_dist_df if col not in sem_dist_word_lst])
red_lev_dist_df = red_lev_dist_df.drop(index=[row for row in lev_dist_df if row not in sem_dist_word_lst])
red_lev_dist_array = red_lev_dist_df.to_numpy()

# the CDI df contains duplicates
counter = Counter(red_lev_dist_df.columns)
duplicates = keys_of_value(counter, 2)

# plot heatmap
fig, ax = plt.subplots(figsize=(40, 30))
sns.heatmap(red_lev_dist_df, cmap="YlGnBu")
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
# ax.set_title('Pairwise Levenshtein distances', size=25)
# ax.tick_params(labelsize=20)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=50)

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
# turn the axis label
for item in ax.get_yticklabels():
    item.set_rotation(0)
# save plot
fig.savefig(save_to + 'phon_red.png', bbox_inches='tight', dpi=200)


""" PART 2: AGE OF ACQUISITION """

# calculate Age of Acquisition differences

AoA_df = pd.DataFrame({})
AoA = items_df["acquisition_age_mean"].tolist()
n = 0

for AoA_1 in AoA:
    AoA_lst = []
    for AoA_2 in AoA:
        AoA_lst.append(abs(AoA_1 - AoA_2))
    AoA_df[n] = AoA_lst
    n += 1

AoA_df.columns = IPA_lst

# Plot heatmap

fig, ax = plt.subplots(figsize=(40, 30))
sns.heatmap(AoA_df, cmap="YlGnBu")
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
# ax.set_title('Pairwise Age of Acquisition differences', size=25)
# ax.tick_params(labelsize=20)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=50)

# ax.set_yticklabels(IPA_lst, rotation=0)
# ax.set_xticklabels(IPA_lst, rotation=90)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
# save plot
fig.savefig(save_to + 'AoA.png', bbox_inches='tight', dpi=200)

lev_dist_df.columns = lev_dist_df.index
lev_dist_array = lev_dist_df.to_numpy()

AoA_array = AoA_df.to_numpy()

""" PART 3: SEMANTIC DISTANCE """

sem_dist_array = sem_dist_df.to_numpy()

# plot heatmap
fig, ax = plt.subplots(figsize=(40, 30))
sns.heatmap(sem_dist_df, cmap="YlGnBu")
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
# ax.set_title('Pairwise Cosine distances', size=0)
# ax.tick_params(labelsize=5)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=50)

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
# turn the axis label
#for item in ax.get_yticklabels():
#    item.set_rotation(0)
# save plot
fig.savefig(save_to + 'sem.png', bbox_inches='tight', dpi=200)

""""""""""""""""""""""""""""""""""""""
""" 1. Sort matrices by their own  """
""""""""""""""""""""""""""""""""""""""

""" <<< levenshtein distance matrix >>> """

""" EUCLIDEAN METRIC """
# sort cols
output_lev_eucl = cluster_mtx(lev_dist_array, metric="euclidean")
conf_mtx_sorted_lev_eucl = output_lev_eucl[0]
col_idc_lev_eucl = output_lev_eucl[1]
# norm cols
norm_conf_mtx_lev_eucl = norm_conf_mtx(lev_dist_array, norm_axis="row")
# plot matrix
fig = plot_conf_mtx(conf_mtx_sorted_lev_eucl, xticklabels=col_idc_lev_eucl, yticklabels=col_idc_lev_eucl, annot=None,
                    plt_title="lev dist - euclidean metric")
# save plot
fig.savefig(save_to + 'phon_eucl.png', bbox_inches='tight', dpi=200)

""" CORRELATION METRIC """
# sort cols
output_lev_corr = cluster_mtx(lev_dist_array, metric="correlation")
conf_mtx_sorted_lev_corr = output_lev_corr[0]
col_idc_lev_corr = output_lev_corr[1]
# norm cols
norm_conf_mtx_lev_corr = norm_conf_mtx(lev_dist_array, norm_axis="row")
# plot matrix
fig = plot_conf_mtx(conf_mtx_sorted_lev_corr, xticklabels=col_idc_lev_corr, yticklabels=col_idc_lev_corr, annot=None,
                    plt_title="lev dist - correlation metric")
# save plot
fig.savefig(save_to + 'phon_corr.png', bbox_inches='tight', dpi=200)

""" <<< reduced levenshtein distance matrix >>> """

""" EUCLIDEAN METRIC """
# sort cols
output_red_lev_eucl = cluster_mtx(red_lev_dist_array, metric="euclidean")
conf_mtx_sorted_red_lev_eucl = output_red_lev_eucl[0]
col_idc_red_lev_eucl = output_red_lev_eucl[1]
# norm cols
norm_conf_mtx_red_lev_eucl = norm_conf_mtx(red_lev_dist_array, norm_axis="row")
# plot matrix
fig = plot_conf_mtx(conf_mtx_sorted_red_lev_eucl, xticklabels=col_idc_red_lev_eucl, yticklabels=col_idc_red_lev_eucl,
                    annot=None, plt_title="lev dist - euclidean metric")
# save plot
fig.savefig(save_to + 'red_phon_eucl.png', bbox_inches='tight', dpi=200)

""" CORRELATION METRIC """
# sort cols
output_red_lev_corr = cluster_mtx(red_lev_dist_array, metric="correlation")
conf_mtx_sorted_red_lev_corr = output_red_lev_corr[0]
col_idc_red_lev_corr = output_red_lev_corr[1]
# norm cols
norm_conf_mtx_red_lev_corr = norm_conf_mtx(red_lev_dist_array, norm_axis="row")
# plot matrix
fig = plot_conf_mtx(conf_mtx_sorted_red_lev_corr, xticklabels=col_idc_red_lev_corr, yticklabels=col_idc_red_lev_corr,
                    annot=None, plt_title="lev dist - correlation metric")
# save plot
fig.savefig(save_to + 'red_phon_corr.png', bbox_inches='tight', dpi=200)

""" << SEMANTIC DISTANCE MATRIX >> """

""" EUCLIDEAN METRIC """
# sort cols
output_sem_eucl = cluster_mtx(sem_dist_array, metric="euclidean")
conf_mtx_sorted_sem_eucl = output_sem_eucl[0]
col_idc_sem_eucl = output_sem_eucl[1]
# norm row
conf_mtx_norm_sem_eucl = norm_conf_mtx(sem_dist_array, norm_axis="row")
# plot matrix
fig = plot_conf_mtx(conf_mtx_sorted_sem_eucl, xticklabels=col_idc_sem_eucl, yticklabels=col_idc_sem_eucl, annot=None,
                    plt_title="semantic dist - euclidean metric")
# save plot
fig.savefig(save_to + 'sem_eucl.png', bbox_inches='tight', dpi=200)

""" CORRELATION METRIC """
# sort cols
output_sem_corr = cluster_mtx(sem_dist_array, metric="correlation")
conf_mtx_sorted_sem_corr = output_sem_corr[0]
col_idc_sem_corr = output_sem_corr[1]
# norm cols
norm_conf_mtx_sem_corr = norm_conf_mtx(sem_dist_array, norm_axis="row")
# plot matrix
fig = plot_conf_mtx(conf_mtx_sorted_sem_corr, xticklabels=col_idc_sem_corr, yticklabels=col_idc_sem_corr, annot=None,
                    plt_title="semantic distance - correlation metric")
# save plot
fig.savefig(save_to + 'sem_corr.png', bbox_inches='tight', dpi=200)

""""""""""""""""""""""""""""""""""""""""""""""""""
""" 2. Sort phon dist matrix by sem dist matrix"""
""""""""""""""""""""""""""""""""""""""""""""""""""

'In order to get information about form-meaning systematicity,' \
'we sort the phonetic distance matrix by the sorting of the semantic distance matrix'

''' euclidean metric '''
# sort matrix
phon_conf_mtx_sem_eucl_sort = red_lev_dist_array[col_idc_sem_eucl]
phon_conf_mtx_sem_eucl_sort = phon_conf_mtx_sem_eucl_sort[:, col_idc_sem_eucl]
# plot matrix
fig = plot_conf_mtx(phon_conf_mtx_sem_eucl_sort, xticklabels=col_idc_sem_eucl, yticklabels=col_idc_sem_eucl, annot=None,
                    plt_title="phonetic distance by semantic distance euclidean sorting")
# save plot
fig.savefig(save_to + 'phon_dist_sorted_by_sem_dist_eucl.png', bbox_inches='tight', dpi=200)

''' correlation metric '''
# sort matrix
phon_conf_mtx_sem_corr_sort = red_lev_dist_array[col_idc_sem_corr]
phon_conf_mtx_sem_corr_sort = phon_conf_mtx_sem_corr_sort[:, col_idc_sem_corr]
# plot matrix
fig = plot_conf_mtx(phon_conf_mtx_sem_corr_sort, xticklabels=col_idc_sem_corr, yticklabels=col_idc_sem_corr,
                    annot=None, plt_title="phonetic distance by semantic distance correlation sorting")
# save plot
fig.savefig(save_to + 'phon_dist_sorted_by_sem_dist_corr.png', bbox_inches='tight', dpi=200)

""""""""""""""""""""""""""""""""""""""""""""""""""
""" 3. Sort sem dist matrix by phon dist matrix"""
""""""""""""""""""""""""""""""""""""""""""""""""""

'In order to get information about form-meaning systematicity,' \
'we sort the semantic distance matrix by the sorting of the phonetic distance matrix'

''' euclidean metric '''
# sort matrix
sem_conf_mtx_phon_eucl_sort = sem_dist_array[col_idc_red_lev_eucl]
sem_conf_mtx_phon_eucl_sort = sem_conf_mtx_phon_eucl_sort[:, col_idc_red_lev_eucl]
# plot matrix
fig = plot_conf_mtx(sem_conf_mtx_phon_eucl_sort, xticklabels=col_idc_red_lev_eucl, yticklabels=col_idc_red_lev_eucl,
                    annot=None, plt_title="semantic distance by phonetic distance (euclidean sorting)")
# save plot
fig.savefig(save_to + 'sem_dist_sorted_by_phon_dist_eucl.png', bbox_inches='tight', dpi=200)

''' correlation metric '''
# sort matrix
sem_conf_mtx_phon_corr_sort = sem_dist_array[col_idc_red_lev_corr]
sem_conf_mtx_phon_corr_sort = sem_conf_mtx_phon_corr_sort[:, col_idc_red_lev_corr]
# plot matrix
fig = plot_conf_mtx(sem_conf_mtx_phon_corr_sort, xticklabels=col_idc_red_lev_corr, yticklabels=col_idc_red_lev_corr,
                    annot=None, plt_title="semantic distance by phonetic distance (correlation sorting)")
# save plot
fig.savefig(save_to + 'sem_dist_sorted_by_phon_dist_corr.png', bbox_inches='tight', dpi=200)

