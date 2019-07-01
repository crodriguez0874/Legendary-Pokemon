# -*- coding: utf-8 -*-
"""
EDA - Summary Statistics

Christian Rodriguez
crodriguez0874@gmail.com
06/20/19

Summary - This script does non-visual data exploration and summary
###statistics.

Finding - Arceus makes up an overwhelming number of the legendaries
(111 of the 650). This is because of the data generation process and all of
its forms. Should consider how to handle this.
"""

###############################################################################
###Loading libraries and data
###############################################################################

import pandas as pd

pokemon = pd.read_csv('data/pokemon_training.csv', encoding='latin-1')

###############################################################################
###Getting to know the data and summary statistics
###############################################################################

###Look at the data
print(pokemon.head())

legendary_pokemon = pokemon[pokemon.is_legendary == 1]
non_legendary_pokemon = pokemon[pokemon.is_legendary == 0]

###Number of legendaries and non-legendaries
print('Number of legendary pokemon: ' + str(len(legendary_pokemon)))
print('Number of non-legendary pokemon: ' + str(len(non_legendary_pokemon)))
print('Percentage of pokemon that are legendary in the training set: '
      + str(round(len(legendary_pokemon) / len(pokemon), 4) * 100 ) + '%')

###Summary Stats
for i in pokemon.columns:
    print('Non-legendary')
    print(non_legendary_pokemon[str(i)].describe())
    print('Legendary')
    print(legendary_pokemon[str(i)].describe())
