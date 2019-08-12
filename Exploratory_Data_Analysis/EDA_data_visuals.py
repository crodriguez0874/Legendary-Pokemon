#-*- coding: utf-8 -*-
"""
EDA - Data Visualizarion

Christian Rodriguez
crodriguez0874@gmail.com
06/20/19

Summary - This script conducts visual univariate and bivariate exploratory data
analysis.

Finding 1 - Of the legendary pokemon, most of them are genderless.

Finding 2 - Legendary pokemon have greater distributions of base stats. Should
consider dimension reduction methods on them (spectral embedding, PCA, kernal
PCA), or aggregate by total (HP + Att. + Sp. Att. + Def. + Sp. Def. + Spd.),
offensive (Att. + Sp. Att. + Spd.), or defensive (HP + Def. + Sp. Def.).

Finding 3 - Weight and Height do not separate the classes vary well. Maybe PCA 
or kernal PCA can better separate the data.
"""

###############################################################################
###Loading libraries and data
###############################################################################

import pandas as pd
import plotnine as p9

pokemon = pd.read_csv('data/pokemon_training.csv', encoding='latin-1')

###############################################################################
###Scatter plot of weight vs height
###############################################################################

scatter_size = (p9.ggplot(data=pokemon,
                          mapping=p9.aes(x='weight_kg',
                                        y='height_m',
                                        color='factor(is_legendary)'))+
      p9.geoms.geom_point(alpha=0.45) +
      p9.labs(title='Pokemon by Size',
              x='Weight (kg)',
              y='Height (m)') +
      p9.scale_color_discrete(name='Class',
                             labels=['Non-legendary', 'Legendary']) +
      p9.theme_bw())
                         

###############################################################################
###Histograms of battle stats between legendaries and non-legendaries
###############################################################################

###HP
prob_den_HP = (p9.ggplot(data=pokemon,
                         mapping=p9.aes(x='hp',
                                        fill='factor(is_legendary)')) +
      p9.geoms.geom_density(alpha=0.5) +
      p9.scale_fill_discrete(name='Class',
                             labels=['Non-legendary', 'Legendary']) +
      p9.labs(title='Probability Densities of HP',
              x='HP') +
      p9.theme_bw())

###Attack
prob_den_att = (p9.ggplot(data=pokemon,
                          mapping=p9.aes(x='attack',
                                         fill='factor(is_legendary)')) +
      p9.geoms.geom_density(alpha=0.5) +
      p9.scale_fill_discrete(name='Class',
                             labels=['Non-legendary', 'Legendary']) +
      p9.labs(title='Probability Densities of Attack',
              x='attack') +
      p9.theme_bw())

###Defense
prob_den_def = (p9.ggplot(data=pokemon,
                          mapping=p9.aes(x='defense',
                                         fill='factor(is_legendary)')) +
      p9.geoms.geom_density(alpha=0.5) +
      p9.scale_fill_discrete(name='Class',
                             labels=['Non-legendary', 'Legendary']) +
      p9.labs(title='Probability Densities of Defense',
              x='defense') +
      p9.theme_bw())

###Sp. Attack
prob_den_spatt = (p9.ggplot(data=pokemon,
                            mapping=p9.aes(x='spattack',
                                           fill='factor(is_legendary)')) +
      p9.geoms.geom_density(alpha=0.5) +
      p9.scale_fill_discrete(name='Class',
                             labels=['Non-legendary', 'Legendary']) +
      p9.labs(title='Probability Densities of Sp. Attack',
              x='sp. attack') +
      p9.theme_bw())

###Sp. Defense
prob_den_spdef = (p9.ggplot(data=pokemon,
                            mapping=p9.aes(x='spdefense',
                                           fill='factor(is_legendary)')) +
      p9.geoms.geom_density(alpha=0.5) +
      p9.scale_fill_discrete(name='Class',
                             labels=['Non-legendary', 'Legendary']) +
      p9.labs(title='Probability Densities of Sp. Defense',
              x='sp. defense') +
      p9.theme_bw())

###Speed
prob_den_speed = (p9.ggplot(data=pokemon,
                            mapping=p9.aes(x='speed',
                                           fill='factor(is_legendary)')) +
      p9.geoms.geom_density(alpha=0.5) +
      p9.scale_fill_discrete(name='Class',
                             labels=['Non-legendary', 'Legendary']) +
      p9.labs(title='Probability Densities of Speed',
              x='speed') +
      p9.theme_bw())

###Total Stats
total_stats = (pokemon.hp + pokemon.attack +
                pokemon.defense + pokemon.spattack +
                pokemon.spdefense + pokemon.speed)
pokemon['total_stats'] = total_stats

prob_den_tot = (p9.ggplot(data=pokemon,
                          mapping=p9.aes(x='total_stats',
                                         fill='factor(is_legendary)')) +
      p9.geoms.geom_density(alpha=0.5) +
      p9.scale_fill_discrete(name='Class',
                             labels=['Non-legendary', 'Legendary']) +
      p9.labs(title='Probability Densities of Total Stats',
              x='total stats') +
      p9.theme_bw())

###############################################################################
###Barplot of Gender by Among legendary and non-legendary pokemon
###############################################################################
      
males = [pokemon['gender'] ==  1]
females = [pokemon['gender'] ==  0]
nogender = [pokemon['gender'].isnull()]

pokemon.iloc[males, pokemon.columns == 'gender'] = 'Male'
pokemon.iloc[nogender, pokemon.columns == 'gender'] = 'No Gender'
pokemon.iloc[females, pokemon.columns == 'gender'] = 'Female'

legendary_pokemon = pokemon[pokemon.is_legendary == 1]
non_legendary_pokemon = pokemon[pokemon.is_legendary == 0]

barplot_L_gender = (p9.ggplot(data=legendary_pokemon,
                              mapping=p9.aes(x='factor(gender)')) +
      p9.geoms.geom_bar(fill='#99FFFF',
                        color='#000000') +
      p9.labs(title='Legendary Pokemon by Gender',
              x='Gender') +
      p9.theme_bw())
      
barplot_nL_gender = (p9.ggplot(data=non_legendary_pokemon,
                               mapping=p9.aes(x='factor(gender)')) +
      p9.geoms.geom_bar(fill='#FF6666',
                        color='#000000') +
      p9.labs(title='Non-Legendary Pokemon by Gender',
              x='Gender') +
      p9.theme_bw())
  
###############################################################################    
###Output
###############################################################################
      
print(scatter_size)
print(prob_den_HP)
print(prob_den_att)
print(prob_den_def)
print(prob_den_spatt)
print(prob_den_spdef)
print(prob_den_speed)
print(prob_den_tot)
print(barplot_L_gender)
print(barplot_nL_gender)
