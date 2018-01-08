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
    

def generation_importance(pokemon, combats):
    generation = pokemon[ ['#','Generation'] ]
    generation_list = pokemon['Generation'].unique().tolist()
    generation_new = [ [[0,0] for x in generation_list] for y in generation_list]
    generate_length = len(generation_list)

    for index, row in combats.head(100).iterrows():
        first = generation[ generation['#']==row['First_pokemon']]['Generation'].values[0] - 1
        second = generation[ generation['#']==row['Second_pokemon']]['Generation'].values[0] - 1
        label = row['Winner']
        if label:
            generation_new[second][first][0] +=1
            generation_new[first][second][1] +=1
        else:
            generation_new[second][first][1] +=1
            generation_new[first][second][0] +=1



    for index, generation_name in enumerate(generation_list):
        generation_tmp = generation_new[index]
        tmp = pd.DataFrame(generation_tmp, columns=['win','lose'])
        tmp.index = generation_list
        tmp.drop(generation_name,inplace=True)
        tmp.plot(kind='bar', title='Generation '+str(generation_name))
        plt.show()

if __name__=='__main__':
    pokemon_data = pd.read_csv('./pokemon.csv')
    combat_data = pd.read_csv('./new_combats.csv')
    #legendary_distribution(pokemon_data['Legendary'])
    #type_distribution(pokemon_data['Type 1'])
    #type_distribution(pokemon_data['Type 2'])
    #legendary_compare(pokemon_data)
    #legendary_importance(pokemon_data, combat_data)
    generation_importance(pokemon_data, combat_data)
    

