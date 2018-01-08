from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score#,ShuffleSplit
#from sklearn.cross_validation import ShuffleSplit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    all_score = list()
    training = list()
    for stas in status:
        #train = feature_matrix[stas].as_matrix()
        #train = [ [x[0]-x[1]] for x in train]
        #score = cross_val_score(rf, train, labels, scoring="r2", cv=ShuffleSplit(n_splits=10, test_size=.1))
        #score = cross_val_score(rf, train, labels, scoring="r2", cv=ShuffleSplit(n=len(feature_matrix.as_matrix()), n_iter=10, test_size=.1))
        train = feature_matrix[stas].as_matrix()
        train = [ [x[0]-x[1]] for x in train]
        training.append(train)
        #all_score.append((round(np.mean(score)), stas))
    #score_df = pd.DataFrame(all_score, columns = ['score', 'feature'])
    #print (score_df)
    tmp = [a+b+c+d+e+f for a,b,c,d,e,f in zip(training[0],training[1],training[2],training[3],training[4],training[5])]
    forest.fit(tmp,labels)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    new_status = list()
    for f in range(len(status)):
        print("%d. feature %s (%f)" % (f + 1, status[indices[f]], importances[indices[f]]))
        new_status.append(status[indices[f]])
        
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(status)), importances[indices],
       color="b", align="center")
    plt.xticks(range(len(status)), new_status)
    plt.xlim([-1, len(status)])
    plt.show()
    

if __name__=='__main__':
    pokemon = pd.read_csv('./pokemon.csv')
    combats = pd.read_csv('./new_combats.csv')
    feature_matrix,labels = status_feature_matrix(pokemon, combats)
    individual_status(feature_matrix, labels)
    
