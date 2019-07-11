# -*- coding: utf-8 -*-
"""
Feature Extraction

Christian Rodriguez
crodriguez0874@gmail.com
06/20/19

Summary - In this script, we try multiple dimension reduction methods on the 
base stats of the pokemon (HP, Attack, Sp. Attack, Defense, Sp. Defense,
Speed). The methods implemented are PCA, polynomial-kernal PCA, RBF-kernal PCA,
Cosine-kernal PCA, and Isomap. Prior to applying the dimension reduction
methods, we made sure to standardize the data via min-max
(x_i - min(x)/max(x)). Also, the parameters for each dimension reduction method
was chosen via grid search and observing which set of parameters best results
in the data being linearly seperable in 3 dimensions. 
"""

###############################################################################
###Loading libraries and data
###############################################################################

import pandas as pd
import sklearn.decomposition as decomp
import sklearn.manifold as mani
import plotnine as p9
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

pokemon = pd.read_csv('data/pokemon_training.csv', encoding='latin-1')

###############################################################################
###Standardizing Features of Interest (min-max)
###############################################################################

stats_features = ['hp', 'attack', 'defense', 'spattack', 'spdefense', 'speed']
pokemon_stats = pokemon.loc[:, stats_features]
pokemon_stats_min = pokemon_stats.min()
pokemon_stats_max = pokemon_stats.max()
pokemon_stats_scaled = pokemon_stats - pokemon_stats_min
pokemon_stats_scaled = pokemon_stats_scaled / pokemon_stats_max

is_legendary = pokemon.loc[:, 'is_legendary']
legendary_index = (is_legendary == 1)
non_legendary_index = (is_legendary == 0)

###############################################################################
###PCA
###############################################################################

###Conducting PCA on the base stats data
basestats_pca = decomp.PCA(n_components=None)
basestats_pca.fit(pokemon_stats_scaled)

basestats_pca_column_names = ['PC'+str(i) for i in
                              range(1, (basestats_pca.n_components_+1))]
basestats_pca_loadings= pd.DataFrame(data=basestats_pca.components_,
                                     index=stats_features,
                                     columns=basestats_pca_column_names)

print(basestats_pca_loadings)

###Base Stats PCA Screeplot
basestats_pca_exp_var = [(sum(basestats_pca.explained_variance_[:i])/
                          sum(basestats_pca.explained_variance_)) for i in
                          range(1, int(basestats_pca.n_components_)+1)]

basestats_screeplot_df = pd.DataFrame(columns=['number_of_components',
                                               'perc_of_var_explained'])
basestats_screeplot_df['number_of_components'] = range(1,7)
basestats_screeplot_df['perc_of_var_explained'] = basestats_pca_exp_var

basestats_pca_screeplot = (p9.ggplot(data=basestats_screeplot_df,
                                     mapping=p9.aes(x='number_of_components',
                                                    y='perc_of_var_explained')) +
                           p9.geoms.geom_point() +
                           p9.geoms.geom_line() +
                           p9.theme_bw() +
                           p9.labs(title='Proportion of Variance Explained (PCA)',
                                   x='Number of Principal Components Used',
                                   y='Percentage of Variance') +
                           p9.ylim(0, 1))
                           
print(basestats_pca_screeplot)
    
###Transforming the base stats data via PCA and projecting to 2D
basestats_PCscores = pd.DataFrame(data=basestats_pca.transform(pokemon_stats_scaled),
                                  columns=['PC1', 'PC2', 'PC3',
                                           'PC4', 'PC5', 'PC6'])
basestats_PCscores['is_legendary'] = is_legendary

basestats_PCscores_plot = (p9.ggplot(data=basestats_PCscores,
                                     mapping=p9.aes(x='PC1',
                                                    y='PC2')) +
                           p9.geoms.geom_point(p9.aes(color='factor(is_legendary)'),
                                               alpha=0.30) +
                           p9.theme_bw() +
                           p9.labs(title='2D Projection of Base Stats via PCA',
                                   x='PC1',
                                   y='PC2') + 
                           p9.scale_color_discrete(name='Class', 
                                                   labels=['Non-legendary',
                                                           'Legendary']))

print(basestats_PCscores_plot)

###3D Representation
fig = plt.figure
ax = plt.axes(projection='3d')
ax.scatter3D(basestats_PCscores.ix[legendary_index , ['PC1']],
             basestats_PCscores.ix[legendary_index , ['PC2']],
             basestats_PCscores.ix[legendary_index , ['PC3']],
             c='b',
             alpha=0.3,
             label='Legendary')
ax.scatter3D(basestats_PCscores.ix[non_legendary_index , ['PC1']],
             basestats_PCscores.ix[non_legendary_index , ['PC2']],
             basestats_PCscores.ix[non_legendary_index , ['PC3']],
             c='r',
             alpha=0.3,
             label='Non-legendary')
ax.set_title('3D Projection of Base Stats via PCA', fontweight='bold')
ax.set_xlabel('PC1', fontweight='bold')
ax.set_ylabel('PC2', fontweight='bold')
ax.set_zlabel('PC3', fontweight='bold')
ax.legend(loc='best', bbox_to_anchor= (-0.25, 0.40, 0.5,0.5))

###############################################################################
###Polynomial-Kernal PCA
###############################################################################

###Conducting polynomial-KPCA (parameters chosen via grid search)
basestats_kpca_poly = decomp.KernelPCA(kernel="poly",
                                       gamma=15,
                                       degree=3)
basestats_kpca_poly.fit(pokemon_stats_scaled)

###Base Stats polynomial-KPCA Screeplot
basestats_kpca_poly_exp_var = [(sum(basestats_kpca_poly.lambdas_[:i])/
                                sum(basestats_kpca_poly.lambdas_)) for i in
                                range(1, len(basestats_kpca_poly.lambdas_)+1)]

basestats_kpca_poly_screeplot_df = pd.DataFrame(columns=['number_of_components',
                                                         'perc_of_var_explained'])
basestats_kpca_poly_screeplot_df['number_of_components'] = range(1,len(basestats_kpca_poly.lambdas_)+1)
basestats_kpca_poly_screeplot_df['perc_of_var_explained'] = basestats_kpca_poly_exp_var

basestats_kpca_poly_screeplot1 = (p9.ggplot(data=basestats_kpca_poly_screeplot_df,
                                 mapping=p9.aes(x='number_of_components',
                                                y='perc_of_var_explained')) +
                                 p9.geoms.geom_point() +
                                 p9.geoms.geom_line() +
                                 p9.theme_bw() +
                                 p9.labs(title='Proportion of Variance Explained (Poly-KPCA)',
                                         x='Number of Principal Components Used',
                                         y='Percentage of Variance') +
                                 p9.ylim(0, 1))

print(basestats_kpca_poly_screeplot1)

###Base Stats polynomial-KPCA Screeplot - a closer look
basestats_kpca_poly_screeplot2 = (basestats_kpca_poly_screeplot1 +
                                  p9.scale_x_continuous(limits=[1,11],
                                                        breaks=range(1,11)))
                                 
print(basestats_kpca_poly_screeplot2)


###Transforming the base stats data via polynomial-KPCA and projecting to 2D
basestats_kpca_poly_column_names = ['PC'+str(i) for i in
                                    range(1, len(basestats_kpca_poly.lambdas_)+1)]
basestats_kPCscores_poly = pd.DataFrame(data=basestats_kpca_poly.transform(pokemon_stats_scaled),
                                        columns=basestats_kpca_poly_column_names)
basestats_kPCscores_poly['is_legendary'] = is_legendary
    
basestats_kPCscores_poly_plot = (p9.ggplot(data=basestats_kPCscores_poly,
                                           mapping=p9.aes(x='PC1',
                                                          y='PC2')) +
                                p9.geoms.geom_point(p9.aes(color='factor(is_legendary)'),
                                                    alpha=0.30) +
                                p9.theme_bw() +
                                p9.labs(title='2D Projection of Base Stats via Poly-KPCA',
                                        x='PC1',
                                        y='PC2') + 
                                p9.scale_color_discrete(name='Class', 
                                                        labels=['Non-legendary',
                                                                'Legendary']))

print(basestats_kPCscores_poly_plot)

###3D Representation
fig = plt.figure
ax = plt.axes(projection='3d')
ax.scatter3D(basestats_kPCscores_poly.ix[legendary_index , ['PC1']],
             basestats_kPCscores_poly.ix[legendary_index , ['PC2']],
             basestats_kPCscores_poly.ix[legendary_index , ['PC3']],
             c='b',
             alpha=0.3,
             label='Legendary')
ax.scatter3D(basestats_kPCscores_poly.ix[non_legendary_index , ['PC1']],
             basestats_kPCscores_poly.ix[non_legendary_index , ['PC2']],
             basestats_kPCscores_poly.ix[non_legendary_index , ['PC3']],
             c='r',
             alpha=0.3,
             label='Non-legendary')
ax.set_title('3D Projection of Base Stats via Poly-KPCA', fontweight='bold')
ax.set_xlabel('PC1', fontweight='bold')
ax.set_ylabel('PC2', fontweight='bold')
ax.set_zlabel('PC3', fontweight='bold')
ax.legend(loc='best', bbox_to_anchor= (-0.25, 0.40, 0.5,0.5))

###############################################################################
###RBF-Kernal PCA
###############################################################################

###Conducting RBF-KPCA (parameters chosen via grid search)
basestats_kpca_RBF = decomp.KernelPCA(kernel="rbf",
                                      gamma=13)
basestats_kpca_RBF.fit(pokemon_stats_scaled)

###Base Stats RBF-KPCA Screeplot
basestats_kpca_RBF_exp_var = [(sum(basestats_kpca_RBF.lambdas_[:i])/
                               sum(basestats_kpca_RBF.lambdas_)) for i in
                               range(1, len(basestats_kpca_RBF.lambdas_)+1)]

basestats_kpca_RBF_screeplot_df = pd.DataFrame(columns=['number_of_components',
                                                        'perc_of_var_explained'])
basestats_kpca_RBF_screeplot_df['number_of_components'] = range(1,len(basestats_kpca_RBF.lambdas_)+1)
basestats_kpca_RBF_screeplot_df['perc_of_var_explained'] = basestats_kpca_RBF_exp_var

basestats_kpca_RBF_screeplot1 = (p9.ggplot(data=basestats_kpca_RBF_screeplot_df,
                                 mapping=p9.aes(x='number_of_components',
                                                y='perc_of_var_explained')) +
                                p9.geoms.geom_point() +
                                p9.geoms.geom_line() +
                                p9.theme_bw() +
                                p9.labs(title='Proportion of Variance Explained (RBF-PCA)',
                                        x='Number of Principal Components Used',
                                        y='Percentage of Variance') +
                                p9.ylim(0, 1))

print(basestats_kpca_RBF_screeplot1)

###Base Stats RBF-KPCA Screeplot - a closer look
basestats_kpca_RBF_screeplot2 = (basestats_kpca_RBF_screeplot1 +
                                 p9.scale_x_continuous(limits=[1,201]))

print(basestats_kpca_RBF_screeplot2)

###Transforming the base stats data via RBF-KPCA and projecting to 2D
basestats_kpca_RBF_column_names = ['PC'+str(i) for i in
                                   range(1, len(basestats_kpca_RBF.lambdas_)+1)]
basestats_kPCscores_RBF = pd.DataFrame(data=basestats_kpca_RBF.transform(pokemon_stats_scaled),
                                       columns=basestats_kpca_RBF_column_names)
basestats_kPCscores_RBF['is_legendary'] = is_legendary
    
basestats_kPCscores_RBF_plot = (p9.ggplot(data=basestats_kPCscores_RBF,
                                      mapping=p9.aes(x='PC1',
                                                     y='PC2')) +
                                p9.geoms.geom_point(p9.aes(color='factor(is_legendary)'),
                                                    alpha=0.30) +
                                p9.theme_bw() +
                                p9.labs(title='2D Projection of Base Stats via RBF-KPCA',
                                        x='PC1',
                                        y='PC2') + 
                                p9.scale_color_discrete(name='Class', 
                                                        labels=['Non-legendary',
                                                                'Legendary']))

print(basestats_kPCscores_RBF_plot)

###3D Representation
fig = plt.figure
ax = plt.axes(projection='3d')
ax.scatter3D(basestats_kPCscores_RBF.ix[legendary_index , ['PC1']],
             basestats_kPCscores_RBF.ix[legendary_index , ['PC2']],
             basestats_kPCscores_RBF.ix[legendary_index , ['PC3']],
             c='b',
             alpha=0.3,
             label='Legendary')
ax.scatter3D(basestats_kPCscores_RBF.ix[non_legendary_index , ['PC1']],
             basestats_kPCscores_RBF.ix[non_legendary_index , ['PC2']],
             basestats_kPCscores_RBF.ix[non_legendary_index , ['PC3']],
             c='r',
             alpha=0.3,
             label='Non-legendary')
ax.set_title('3D Projection of Base Stats via RBF-KPCA', fontweight='bold')
ax.set_xlabel('PC1', fontweight='bold')
ax.set_ylabel('PC2', fontweight='bold')
ax.set_zlabel('PC3', fontweight='bold')
ax.legend(loc='best', bbox_to_anchor= (-0.25, 0.40, 0.5,0.5))

###############################################################################
###Cosine-Kernal PCA
###############################################################################

###Conducting Cosine-KPCA (parameters chosen via grid search)
basestats_kpca_cosine = decomp.KernelPCA(kernel="cosine",
                                         gamma=15)
basestats_kpca_cosine.fit(pokemon_stats_scaled)

###Base Stats Cosine-KPCA Screeplot
basestats_kpca_cosine_exp_var = [(sum(basestats_kpca_cosine.lambdas_[:i])/
                                  sum(basestats_kpca_cosine.lambdas_)) for i in
                                  range(1, len(basestats_kpca_cosine.lambdas_)+1)]

basestats_kpca_cosine_screeplot_df = pd.DataFrame(columns=['number_of_components',
                                                           'perc_of_var_explained'])
basestats_kpca_cosine_screeplot_df['number_of_components'] = range(1,len(basestats_kpca_cosine.lambdas_)+1)
basestats_kpca_cosine_screeplot_df['perc_of_var_explained'] = basestats_kpca_cosine_exp_var

basestats_kpca_cosine_screeplot1 = (p9.ggplot(data=basestats_kpca_cosine_screeplot_df,
                                              mapping=p9.aes(x='number_of_components',
                                                             y='perc_of_var_explained')) +
                                   p9.geoms.geom_point() +
                                   p9.geoms.geom_line() +
                                   p9.theme_bw() +
                                   p9.labs(title='Proportion of Variance Explained (Cosine-PCA)',
                                           x='Number of Principal Components Used',
                                           y='Percentage of Variance') +
                                   p9.ylim(0, 1))

print(basestats_kpca_cosine_screeplot1)

###Base Stats Cosine-KPCA Screeplot - a closer look
basestats_kpca_cosine_screeplot2 = (basestats_kpca_cosine_screeplot1 +
                                    p9.scale_x_continuous(limits=[1,11],
                                                          breaks=range(1,11)))

print(basestats_kpca_cosine_screeplot2)

###Transforming the base stats data via Cosine-KPCA and projecting to 2D
basestats_kpca_cosine_column_names = ['PC'+str(i) for i in
                                      range(1, len(basestats_kpca_cosine.lambdas_)+1)]
basestats_kPCscores_cosine = pd.DataFrame(data=basestats_kpca_cosine.transform(pokemon_stats_scaled),
                                          columns=basestats_kpca_cosine_column_names)
basestats_kPCscores_cosine['is_legendary'] = is_legendary
    
basestats_kPCscores_cosine_plot = (p9.ggplot(data=basestats_kPCscores_cosine,
                                      mapping=p9.aes(x='PC1',
                                                     y='PC2')) +
                                   p9.geoms.geom_point(p9.aes(color='factor(is_legendary)'),
                                                       alpha=0.30) +
                                   p9.theme_bw() +
                                   p9.labs(title='2D Projection of Base Stats via Cosine-KPCA',
                                        x='PC1',
                                        y='PC2') + 
                                   p9.scale_color_discrete(name='Class', 
                                                           labels=['Non-legendary',
                                                                   'Legendary']))

print(basestats_kPCscores_cosine_plot)

###3D Representation
fig = plt.figure
ax = plt.axes(projection='3d')
ax.scatter3D(basestats_kPCscores_cosine.ix[legendary_index , ['PC1']],
             basestats_kPCscores_cosine.ix[legendary_index , ['PC2']],
             basestats_kPCscores_cosine.ix[legendary_index , ['PC3']],
             c='b',
             alpha=0.3,
             label='Legendary')
ax.scatter3D(basestats_kPCscores_cosine.ix[non_legendary_index , ['PC1']],
             basestats_kPCscores_cosine.ix[non_legendary_index , ['PC2']],
             basestats_kPCscores_cosine.ix[non_legendary_index , ['PC3']],
             c='r',
             alpha=0.3,
             label='Non-Legendary')
ax.set_title('3D Projection of Base Stats via Cosine-KPCA', fontweight='bold')
ax.set_xlabel('PC1', fontweight='bold')
ax.set_ylabel('PC2', fontweight='bold')
ax.set_zlabel('PC3', fontweight='bold')
ax.legend(loc='best', bbox_to_anchor= (-0.25, 0.40, 0.5,0.5))

###############################################################################
###Isomap
###############################################################################

###Conducting the isomapping (parameters chosen via grid search)
basestats_isomap = mani.Isomap(n_components=3, n_neighbors=3)
basestats_isomap.fit(pokemon_stats_scaled)

###Projecting the data to 2D and plotting
basestats_isomap_df = pd.DataFrame(data=basestats_isomap.transform(pokemon_stats_scaled),
                                   columns=['1D', '2D', '3D'])
basestats_isomap_df['is_legendary'] = is_legendary

basestats_isomap_plot = (p9.ggplot(data=basestats_isomap_df,
                                   mapping=p9.aes(x='1D',
                                                  y='2D')) +
                         p9.geoms.geom_point(p9.aes(color='factor(is_legendary)'),
                                             alpha=0.30) +
                         p9.theme_bw() +
                         p9.scale_color_discrete(name='Class',
                                                 labels=['Non-legendary',
                                                         'Legendary']) +
                         p9.labs(title='2D Projection of Base Stats via Isomap'))

print(basestats_isomap_plot)

###3D Representation
fig = plt.figure
ax = plt.axes(projection='3d')
ax.scatter3D(basestats_isomap_df.ix[legendary_index , ['1D']],
             basestats_isomap_df.ix[legendary_index , ['2D']],
             basestats_isomap_df.ix[legendary_index , ['3D']],
             c='b',
             alpha=0.3,
             label='Legendary')
ax.scatter3D(basestats_isomap_df.ix[non_legendary_index , ['1D']],
             basestats_isomap_df.ix[non_legendary_index , ['2D']],
             basestats_isomap_df.ix[non_legendary_index , ['3D']],
             c='r',
             alpha=0.3,
             label='Non-legendary')
ax.set_title('3D Projection of Base Stats via Isomap', fontweight='bold')
ax.set_xlabel('1D', fontweight='bold')
ax.set_ylabel('2D', fontweight='bold')
ax.set_zlabel('3D', fontweight='bold')
ax.legend(loc='best', bbox_to_anchor= (-0.25, 0.40, 0.5,0.5))
