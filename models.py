from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import pickle


def one_hot_encoding(data):
    type_list = data.unique().tolist()
    type_list = [ [x] for x in type_list]
    mlb = MultiLabelBinarizer()
    mlb.fit(type_list)
    targets = data.values
    targets = [ [a] for a in targets]
    feature = mlb.transform(targets)
    return feature


def get_feature_matrix_onehot(pokemon, combats):
    type_feature = one_hot_encoding(pokemon['Type 1'])
    generation_feature = one_hot_encoding(pokemon['Generation'])
    speed_feature = MinMaxScaler().fit_transform([[a] for a in pokemon['Speed'].tolist()])
    speed_feature = [round(float(a[0]),3) for a in speed_feature]
    legendary_feature = pokemon['Legendary'].tolist()
    legendary_feature = [int(x==True) for x in legendary_feature]
    training = list()
    labels = list()
    for index, row in combats.iterrows():
        first_index = row['First_pokemon']-1
        second_index = row['Second_pokemon']-1
        first_feature = type_feature[first_index].tolist() + [speed_feature[first_index]] + generation_feature[first_index].tolist() + [legendary_feature[first_index]]
        second_feature =  type_feature[second_index].tolist() + [speed_feature[second_index]] + generation_feature[second_index].tolist() + [legendary_feature[second_index]]
        final_feature = first_feature + second_feature
        label = row['Winner']
        training.append(final_feature)
        labels.append(label)
        
    return training, labels

def get_feature_matrix(pokemon, combats):
    type_feature = pokemon['Type 1'].tolist()
    type_list = pokemon['Type 1'].unique().tolist()
    type_feature = [type_list.index(x) for x in type_feature]
    generation_feature = pokemon['Generation'].tolist()
    speed_feature = MinMaxScaler().fit_transform([[a] for a in pokemon['Speed'].tolist()])
    speed_feature = [round(float(a[0]),3) for a in speed_feature]
    legendary_feature = pokemon['Legendary'].tolist()
    legendary_feature = [int(x==True) for x in legendary_feature]
    training = list()
    labels = list()
    for index, row in combats.iterrows():
        first_index = row['First_pokemon']-1
        second_index = row['Second_pokemon']-1
        first_feature = [type_feature[first_index]] + [speed_feature[first_index]] + [generation_feature[first_index]] + [legendary_feature[first_index]]
        second_feature =  [type_feature[second_index]] + [speed_feature[second_index]] + [generation_feature[second_index]] + [legendary_feature[second_index]]
        final_feature = first_feature + second_feature
        label = row['Winner']
        training.append(final_feature)
        labels.append(label)
        
    return training, labels


def model(pokemon,combats):
    training, labels = get_feature_matrix(pokemon, combats)
    training = np.array(training)
    labels = np.array(labels)
    svm_clf = SVC()
    rf_clf = RandomForestClassifier()
    ada_clf = AdaBoostClassifier()
    gbdt_clf = GradientBoostingClassifier()
    kf = KFold(n_splits=2)
    all_scores = list()
    for train_index, test_index in kf.split(training):
        X_train, X_test = training[train_index], training[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]
        count = 0        
        for clf,name in [(svm_clf,'svm'),(rf_clf,'rf'),(ada_clf,'ada'),(gbdt_clf,'gbdt')]:
            clf.fit(X_train, Y_train)
            y_pred = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                prob_pos = clf.predict_proba(X_test)[:, 1]
            else:  
                prob_pos = clf.decision_function(X_test)
                prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            scores= [precision_score(Y_test, y_pred), recall_score(Y_test, y_pred), f1_score(Y_test, y_pred), roc_auc_score(Y_test, prob_pos), name]
            all_scores.append(scores)
            print (scores)
    print (all_scores)
    pickle.dump(all_scores, open('scores.pickle','wb'))
    
def model_onehot(pokemon,combats):
    training, labels = get_feature_matrix_onehot(pokemon, combats)
    training = np.array(training)
    labels = np.array(labels)
    svm_clf = SVC()
    rf_clf = RandomForestClassifier()
    ada_clf = AdaBoostClassifier()
    gbdt_clf = GradientBoostingClassifier()
    kf = KFold(n_splits=2)
    all_scores = list()
    for train_index, test_index in kf.split(training):
        X_train, X_test = training[train_index], training[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]
        count = 0        
        for clf,name in [(svm_clf,'svm'),(rf_clf,'rf'),(ada_clf,'ada'),(gbdt_clf,'gbdt')]:
            clf.fit(X_train, Y_train)
            y_pred = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                prob_pos = clf.predict_proba(X_test)[:, 1]
            else:  
                prob_pos = clf.decision_function(X_test)
                prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            scores= [precision_score(Y_test, y_pred), recall_score(Y_test, y_pred), f1_score(Y_test, y_pred), roc_auc_score(Y_test, prob_pos), name]
            all_scores.append(scores)
            print (scores)
    print (all_scores)
    pickle.dump(all_scores, open('scores_onehot.pickle','wb'))
    

if __name__=='__main__':
    pokemon = pd.read_csv('./pokemon.csv')
    combats = pd.read_csv('./new_combats.csv')
    #one_hot_encoding(pokemon['Type 1'])
    #training, labels = get_feature_matrix_onehot(pokemon, combats)
    #training, labels = get_feature_matrix(pokemon, combats)
    model(pokemon, combats)
    model_onehot(pokemon, combats)
