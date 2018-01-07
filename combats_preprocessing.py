import pandas as pd
import os


#0: first win, 1: second win
def change_winner2int(combats):
    for index, row in combats.iterrows():
        if row['First_pokemon']==row['Winner']:
            row['Winner'] = 0
        elif row['Second_pokemon']==row['Winner']:
            row['Winner'] = 1
        else:
            print ('No winner')
    combats.to_csv('./new_combats.csv', index=False)



if __name__=='__main__':
    if not os.path.isdir('./new_combats.csv'):
        change_winner2int(pd.read_csv('./combats.csv'))
                         
