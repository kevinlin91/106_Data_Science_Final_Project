import pandas as pd
import matplotlib.pyplot as plt


def legendary_distribution(Legendary):
    legend_count = Legendary.value_counts()
    legend_count.plot(kind='bar')    
    plt.show()

def legendary_compare(pokemon):
    legendary = pokemon[ pokemon['Legendary'] == True][ ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'] ]
    non_legendary = pokemon[ pokemon['Legendary'] == False][ ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'] ]
    index = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    combine = pd.concat([legendary.mean(), non_legendary.mean()], axis=1)
    combine.columns = ['legendary', 'non_legendary']
    combine.plot(kind='bar')
    plt.show()

def legendary_importance(pokemon,combats):
    legendary_id = pokemon[ pokemon['Legendary'] == True]['#'].tolist()
    legendary_win = 0
    legendary_lose = 0
    both = 0
    for index, row in combats.iterrows():
        if row['First_pokemon'] in legendary_id and not row['Second_pokemon'] in legendary_id:
            if row['Winner']==0:
                legendary_win +=1
            else:
                legendary_lose +=1
        elif not row['First_pokemon'] in legendary_id and row['Second_pokemon'] in legendary_id:
            if row['Winner']==1:
                legendary_win +=1
            else:
                legendary_lose +=0
            
        elif row['First_pokemon'] in legendary_id and row['Second_pokemon'] in legendary_id:
            both +=1
    tmp = pd.DataFrame({'count':[legendary_win,legendary_lose], 'Result':['win','lose']})
    tmp.plot(kind='bar', x='Result', y='count', title='legendary battle with others')
    plt.show()

def type_distribution(_type):
    type_count = _type.value_counts()
    type_count.plot(kind='bar')
    plt.show()
    

def generation_importance(pokemon, combats):
    generation = pokemon[ ['#','Generation'] ]
    generation_new = [  [ [0,0],[0,0],[0,0],[0,0],[0,0],[0,0] ],
                        [ [0,0],[0,0],[0,0],[0,0],[0,0],[0,0] ],
                        [ [0,0],[0,0],[0,0],[0,0],[0,0],[0,0] ],
                        [ [0,0],[0,0],[0,0],[0,0],[0,0],[0,0] ],
                        [ [0,0],[0,0],[0,0],[0,0],[0,0],[0,0] ],
                        [ [0,0],[0,0],[0,0],[0,0],[0,0],[0,0] ] ]
    
    for index, row in combats.head(1000).iterrows():
        first = generation[ generation['#']==row['First_pokemon']]['Generation'].values[0] - 1
        second = generation[ generation['#']==row['First_pokemon']]['Generation'].values[0] - 1
        label = row['Winner']
        if label:
            generation_new[second][first][0] +=1
            generation_new[first][second][1] +=1
        else:
            generation_new[second][first][1] +=1
            generation_new[first][second][0] +=1
    
    df_generation_1 = pd.DataFrame({'1':generation_new[0][0], '2':generation_new[0][1], '3':generation_new[0][2], '4':generation_new[0][3], '5':generation_new[0][4], '6':generation_new[0][5]})
    df_generation_1.index = ['Win', 'Lose']
    df_generation_2 = pd.DataFrame({'1':generation_new[1][0], '2':generation_new[1][1], '3':generation_new[1][2], '4':generation_new[1][3], '5':generation_new[1][4], '6':generation_new[1][5]})
    df_generation_2.index = ['Win', 'Lose']
    df_generation_3 = pd.DataFrame({'1':generation_new[2][0], '2':generation_new[2][1], '3':generation_new[2][2], '4':generation_new[2][3], '5':generation_new[2][4], '6':generation_new[2][5]})
    df_generation_3.index = ['Win', 'Lose']
    df_generation_4 = pd.DataFrame({'1':generation_new[3][0], '2':generation_new[3][1], '3':generation_new[3][2], '4':generation_new[3][3], '5':generation_new[3][4], '6':generation_new[3][5]})
    df_generation_4.index = ['Win', 'Lose']
    df_generation_5 = pd.DataFrame({'1':generation_new[4][0], '2':generation_new[4][1], '3':generation_new[4][2], '4':generation_new[4][3], '5':generation_new[4][4], '6':generation_new[4][5]})
    df_generation_5.index = ['Win', 'Lose']
    df_generation_6 = pd.DataFrame({'1':generation_new[5][0], '2':generation_new[5][1], '3':generation_new[5][2], '4':generation_new[5][3], '5':generation_new[5][4], '6':generation_new[5][5]})
    df_generation_6.index = ['Win', 'Lose']
    df_generation_1.plot(kind='bar', title='First_Generation')
    plt.show()
    df_generation_2.plot(kind='bar', title='Second_Generation')
    plt.show()
    df_generation_3.plot(kind='bar', title='Third_Generation')
    plt.show()
    df_generation_4.plot(kind='bar', title='Fourth_Generation')
    plt.show()
    df_generation_5.plot(kind='bar', title='Fifth_Generation')
    plt.show()
    df_generation_6.plot(kind='bar', title='Sixth_Generation')
    plt.show()
    

if __name__=='__main__':
    pokemon_data = pd.read_csv('./pokemon.csv')
    combat_data = pd.read_csv('./new_combats.csv')
    #legendary_distribution(pokemon_data['Legendary'])
    type_distribution(pokemon_data['Type 1'])
    #type_distribution(pokemon_data['Type 2'])
    #legendary_compare(pokemon_data)
    #legendary_importance(pokemon_data, combat_data)
    #generation_importance(pokemon_data, combat_data)
    

