from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score,ShuffleSplit
#from sklearn.cross_validation import 
import pandas as pd
import numpy as np


def status_feature_matrix(pokemon, combats):
    status = pokemon[ ['#', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
    feature_matrix = pd.DataFrame()
    labels = list()
    for index, row in combats.iterrows():
        first = status[ status['#']==row['First_pokemon']][ ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'] ]
        first.index = [index]
        second = status[ status['#']==row['Second_pokemon']][ ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'] ]
        second.index = [index]
        feature = pd.concat([first,second],axis=1)
        label = row['Winner']
        feature_matrix = feature_matrix.append(feature)
        labels.append(label)
    return feature_matrix,labels


def individual_status(feature_matrix,labels):
    status = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    rf = RandomForestRegressor(n_estimators=20, max_depth=4)
    all_score = list()
    for stas in status:
        score = cross_val_score(rf, feature_matrix[ stas ].as_matrix(), labels, scoring="r2", cv=ShuffleSplit(n_splits=10, test_size=.1))
        all_score.append((round(np.mean(score)), stas))
    score_df = pd.DataFrame(all_score, columns = ['score', 'feature'])
    print (score_df)
    

if __name__=='__main__':
    pokemon = pd.read_csv('./pokemon.csv')
    combats = pd.read_csv('./new_combats.csv')
    feature_matrix,labels = status_feature_matrix(pokemon, combats)
    individual_status(feature_matrix, labels)
    
