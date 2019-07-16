# -*- coding: utf-8 -*-
"""
Feature Extraction TPP

Christian Rodriguez
crodriguez0874@gmail.com
07/12/19

Summary - This is an identical script to that of Feature_Extraction.py in the
Feature Extraction directory. Here, we simply omitted the graphics since 
we will be re-using the code here to fit PCA, the kernal PCAs, and isomapping. 
"""

###############################################################################
###Loading libraries and data
###############################################################################

import pandas as pd
import sklearn.decomposition as decomp
import sklearn.manifold as mani

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

###Transforming the base stats data via PCA and projecting to 2D
basestats_PCscores = pd.DataFrame(data=basestats_pca.transform(pokemon_stats_scaled),
                                  columns=[basestats_pca_column_names])

###############################################################################
###Polynomial-Kernal PCA
###############################################################################

###Conducting polynomial-KPCA (parameters chosen via grid search)
basestats_kpca_poly = decomp.KernelPCA(kernel="poly",
                                       gamma=15,
                                       degree=3)
basestats_kpca_poly.fit(pokemon_stats_scaled)

###Transforming the base stats data via polynomial-KPCA and projecting to 2D
basestats_kpca_poly_column_names = ['PC'+str(i) for i in
                                    range(1, len(basestats_kpca_poly.lambdas_)+1)]
basestats_kPCscores_poly = pd.DataFrame(data=basestats_kpca_poly.transform(pokemon_stats_scaled),
                                        columns=basestats_kpca_poly_column_names)

###############################################################################
###RBF-Kernal PCA
###############################################################################

###Conducting RBF-KPCA (parameters chosen via grid search)
basestats_kpca_RBF = decomp.KernelPCA(kernel="rbf",
                                      gamma=13)
basestats_kpca_RBF.fit(pokemon_stats_scaled)

###Transforming the base stats data via RBF-KPCA and projecting to 2D
basestats_kpca_RBF_column_names = ['PC'+str(i) for i in
                                   range(1, len(basestats_kpca_RBF.lambdas_)+1)]
basestats_kPCscores_RBF = pd.DataFrame(data=basestats_kpca_RBF.transform(pokemon_stats_scaled),
                                       columns=basestats_kpca_RBF_column_names)

###############################################################################
###Cosine-Kernal PCA
###############################################################################

###Conducting Cosine-KPCA (parameters chosen via grid search)
basestats_kpca_cosine = decomp.KernelPCA(kernel="cosine",
                                         gamma=15)
basestats_kpca_cosine.fit(pokemon_stats_scaled)

###Transforming the base stats data via Cosine-KPCA and projecting to 2D
basestats_kpca_cosine_column_names = ['PC'+str(i) for i in
                                      range(1, len(basestats_kpca_cosine.lambdas_)+1)]
basestats_kPCscores_cosine = pd.DataFrame(data=basestats_kpca_cosine.transform(pokemon_stats_scaled),
                                          columns=basestats_kpca_cosine_column_names)

###############################################################################
###Isomap
###############################################################################

###Conducting the isomapping (parameters chosen via grid search)
basestats_isomap = mani.Isomap(n_components=3, n_neighbors=3)
basestats_isomap.fit(pokemon_stats_scaled)

###Projecting the data to 2D and plotting
basestats_isomap_df = pd.DataFrame(data=basestats_isomap.transform(pokemon_stats_scaled),
                                   columns=['1D', '2D', '3D'])
