import pandas as pd
save_to_csv = './csv/'

item_data = pd.read_csv("./item_data.csv", header=int(), index_col=0)

mean_aoa = pd.DataFrame(columns=['word', 'mean_aoa'])

for row in item_data.index:
    row_lst = item_data.loc[row, :].values.tolist()
    new_lst = []
    for item in row_lst[2:]:
        if item < 0.5:
            new_lst.append(item)
    mean_aoa.loc[len(mean_aoa)] = [row, len(new_lst) + 16]

mean_aoa = mean_aoa.set_index('word')

mean_aoa.to_csv(save_to_csv + 'mean_aoa.csv')