import pandas as pd
import matplotlib.pyplot as plt


def type_distribution(_type):
    type_count = _type.value_counts()
    type_count.plot(kind='bar')
    plt.show()

def type_combat(pokemon, combats):
    type_pokemon = pokemon[ ['#', 'Type 1'] ]
    type_list = pokemon['Type 1'].unique().tolist()
    type_length = len(type_list)
    type_combat = [ [[0,0] for x in type_list] for y in type_list]
    for index,row in combats.iterrows():
        first_type = type_pokemon[ type_pokemon['#']==row['First_pokemon']]['Type 1'].values[0]
        second_type = type_pokemon[ type_pokemon['#']==row['Second_pokemon']]['Type 1'].values[0]
        first_type_index = type_list.index(first_type)
        second_type_index = type_list.index(second_type)
        label = row['Winner']
        if label:
            type_combat[second_type_index][first_type_index][0] +=1
            type_combat[first_type_index][second_type_index][1] +=1
        else:
            type_combat[second_type_index][first_type_index][1] +=1
            type_combat[first_type_index][second_type_index][0] +=1
    for index, type_name in enumerate(type_list):
        type_tmp = type_combat[index]
        tmp = pd.DataFrame(type_tmp, columns=['win','lose'])
        tmp.index = type_list
        tmp.drop(type_name,inplace=True)
        tmp.plot(kind='bar',title=type_name)
        plt.show()
    
        



if __name__=='__main__':
    pokemon_data = pd.read_csv('./pokemon.csv')
    combat_data = pd.read_csv('./new_combats.csv')
    #type_distribution(pokemon_data['Type 1'])
    #type_distribution(pokemon_data['Type 2'])
    type_combat(pokemon_data, combat_data)
