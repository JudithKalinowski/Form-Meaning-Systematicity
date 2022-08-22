import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
from mycolorpy import colorlist as mcp

NOR_X = pd.read_csv("./NOR_X.csv", header=int(), index_col=0, sep=";")
NOR_y = pd.read_csv("./NOR_y.csv", header=int(), index_col=0, sep=";")
word_df = pd.read_csv("./word_df.csv", header=int(), index_col=0)
lev_dist_df = pd.read_csv("./pairwise_lev_dist_df.csv", header=int(), index_col=0)
sem_dist_df = pd.read_csv("./sem_dist/cos_sim_df.csv", header=int(), index_col=0)

# define where to save plots and csv files
save_to_plots = './plots/'
save_to_csv = './csv/'

# define colors for plots
cm = plt.cm.get_cmap('YlGnBu')  # colormap
colors = mcp.gen_color(cmap="YlGnBu", n=5)  # get colors from color palette

# store word items from the CDI to a list/df
items_lst = word_df.index.tolist()   # list of word IDs
items_lst_engl = word_df['Translation'].tolist()   # list of english translations of words

# reduce lev dist lst/df due to less words in the sem dist file
sem_dist_word_lst = sem_dist_df.columns     # words in the sem dist matrix
CDI_word_lst = word_df.definition_new.to_list()     # words in the df we are using
lev_dist_df.columns = CDI_word_lst
lev_dist_df.index = CDI_word_lst
reduced_lev_dist_df = lev_dist_df.drop(columns=[col for col in lev_dist_df if col not in sem_dist_word_lst])
reduced_lev_dist_df = reduced_lev_dist_df.drop(index=[row for row in lev_dist_df if row not in sem_dist_word_lst])
reduced_lev_dist_array = reduced_lev_dist_df.to_numpy()

# get reduced list of translations so that they fit with the semantic distance data
engl_word_lst_red = []
for i in word_df.index:
    if word_df.at[i, 'definition_new'] in sem_dist_word_lst:
        engl_word_lst_red.append(word_df.at[i, 'Translation'])
id_to_str = pd.DataFrame(word_df, columns=['num_item_id', 'Translation', 'IPA'])
id_to_str = id_to_str.set_index(['num_item_id'])    # dataframe with word ID, translation of the word and IPA


''' in the following, we work on the IPA transcriptions of the IPA dataframe
so that we can use it for the levenshtein distance '''

# preparation of dataframes
lst_index_id = items_lst + ['age', 'gap between observations', 'vocabulary size']
lst_index_ipa = lev_dist_df.columns.tolist()
lst_index_ipa_full = lev_dist_df.columns.tolist() + ['age', 'gap between observations', 'vocabulary size']
lev_dist_df['index'] = lst_index_ipa
lev_dist_df = lev_dist_df.set_index('index')

# the CDI df contains duplicates
counter = Counter(reduced_lev_dist_df.columns)
def keys_of_value(d, value):
    return [key for key, val in d.items() if val == value]

duplicates = keys_of_value(counter, 2)
sem_dist_array = sem_dist_df.to_numpy()

''' we will now plot all the words' pairwise semantic distance by their phonetic distance and, based on these data, 
do a linear regression. We will store the slopes of the regression lines in a df to use them for another plot later '''

#create df to save the slopes
slope_df = pd.DataFrame(columns=['word', 'slope'])

for word, transl in zip(sem_dist_word_lst[600:], engl_word_lst_red[600:]):
    plt.figure(figsize=(6, 6), dpi=200)
    word = word.split('.')[0]

    if isinstance(reduced_lev_dist_df[word], pd.DataFrame):
        # define x, y and drop row with word=word because we do not need distances between identical words
        x = reduced_lev_dist_df[word].iloc[:, 0].drop(word)
        y = sem_dist_df[word].drop(word)
        # get coeffs of linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        ax = sns.regplot(x, y,
                         ci=None, scatter_kws={"color": colors[2]},
                         line_kws={"color": colors[3], 'label': "y={0:.1f}x+{1:.1f}".format(slope, intercept)},
                         marker='+')
    else:
        # define x, y and drop row with word=word because we do not need distances between identical words
        x = reduced_lev_dist_df[word].drop(word)
        y = sem_dist_df[word].drop(word)
        # get coeffs of linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        ax = sns.regplot(x, y, ci=None, scatter_kws={"color": colors[2]},
                         line_kws={"color": colors[3], 'label': "y={0:.1f}x+{1:.1f}".format(slope, intercept)},
                         marker='+')
    # plot legend
    #ax.legend()    # uncomment if you want to add the function of the regression line + R² and p-value
    plt.title("%s" % transl, size=15)
    plt.xlabel('phonetic distance', size=18)
    plt.xticks(fontsize=15)
    plt.ylabel('semantic distance', size=18)
    plt.yticks(fontsize=15)
    plt.savefig(save_to_plots + 'sem_phon_dist/' + '%s.png' % word, format="png")
    slope_df.loc[len(slope_df.index)] = [word, slope]

# get original item id for slopes
reduced_word_id = []
reduced_word_letters = []
for i in word_df.index:
    if word_df.at[i, 'definition_new'] in sem_dist_word_lst:
        reduced_word_id.append(i)
        reduced_word_letters.append(word_df.at[i, 'definition_new'])
slope_df['item_id'] = reduced_word_id
slope_df = slope_df.set_index('item_id')
slope_df.to_csv(save_to_csv + 'sem_phon_dist.csv')

""" Plot all slopes by mean AoA of words """
# load slope df if you do not want to run all the calculations above
slope_df = pd.read_csv("./csv/sem_phon_dist.csv", index_col=0)
# get mean AoA
aoa_lst = []
for item in slope_df.index:
    if item in word_df.index:
        aoa_lst.append(word_df.at[item, 'acquisition_age_mean'])

# get coeffs of linear fit; uncomment if you want to add the regression line w/ R² and p-value
#slope, intercept, r_value, p_value, std_err = stats.linregress(x=aoa_lst, y=slope_df.slope)
#ax = sns.regplot(x=aoa_lst, y=slope_df.slope, x_ci='ci', fit_reg=True, ci=95, scatter_kws={"color": "green"},
                         #line_kws={"color": "royalblue",
                                   #'label': "$y={0:.3f}x{1:.2f}$, $R^2={2:.2f}$, $p={3:.2f}$".format(slope, intercept, r_value**2, p_value)},
                         #marker='o', alpha=0.5)
plt.figure(figsize=(8, 5), dpi=200)
plt.grid(color=colors[2], linestyle='--', linewidth=0.5, alpha=0.4)
plt.axhline(y=0, color=colors[3], linewidth=1, alpha=0.4)
ax = plt.scatter(x=aoa_lst, y=slope_df.slope, marker='o', alpha=0.4, s=100, color=colors[3])
# plot legend
#ax.legend()    # uncomment if you want to add the function of the regression line + R² and p-value
#plt.title("Systematicity of words by mean AoA", size=12)
plt.xlabel('mean AoA (in months)', size=15)
plt.xticks(fontsize=12)
plt.ylabel('slope (measure for word systematicity)', size=15)
plt.yticks(fontsize=12)
plt.savefig(save_to_plots + 'sem_phon_dist/summary/' + 'Slopes_Summary_AoA.png', format="png", dpi=500)


''' check how many slopes are positive/negative etc '''
slope_df_sorted = slope_df.sort_values('slope')
# Plot histogram
fig, ax = plt.subplots()
ax.margins(0)
ax.axhspan(ymin=0, ymax=100, xmin=0, xmax=0.375, facecolor=colors[1], alpha=0.5)    # background color
ax.axhspan(ymin=0, ymax=100, xmin=0.375, xmax=1, facecolor=colors[2], alpha=0.5)    # background color
entries, bin_edges, c = plt.hist(slope_df_sorted.slope, color=colors[3], edgecolor='black', bins=[-0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015,
                                      0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05])
plt.text(-0.028, 90, '243 words', fontsize=13)
plt.text(0.032, 90, '418 words', fontsize=13)
#plt.title('number of words by slope')  # uncomment if you want to add a title to your plot
plt.xlabel('slope (measure for word systematicity)', fontsize=15)
plt.xticks(fontsize=12)
plt.ylabel('number of words', fontsize=15)
plt.yticks(fontsize=12)
plt.savefig(save_to_plots + 'sem_phon_dist/summary/' + 'number_of_words_by_slope.png', format="png", dpi=500)

# get the number of words which are negative systematic and positive systematic respectively
print('Negative Systematic Words:', sum(entries[0:6]))
print('Positive Systematic Words:', sum(entries[6:]))