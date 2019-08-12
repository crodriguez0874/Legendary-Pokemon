# -*- coding: utf-8 -*-
"""
Tuning and Predictions Preparation

Christian Rodriguez
crodriguez0874@gmail.com
07/15/19

Summary - This script generates the 8 pairs of training and test sets that will
be used in our prediction models. We use an edited version of
Feature_Extraction.py here to save us some work.
"""
###############################################################################
###Loading libraries and data
###############################################################################

import pandas as pd
import Feature_Extraction_TPP as FeatExt

training = pd.read_csv('data/pokemon_training.csv', encoding='latin-1')
test = pd.read_csv('data/pokemon_test.csv', encoding='latin-1')

###############################################################################
###Preparation
###############################################################################

column_names1 = ['hp', 'attack', 'defense', 'spattack', 'spdefense', 'speed']
column_names2 = ['is_legendary', 'weight_kg', 'height_m', 'gender']

training_males = training['gender'] ==  1
training_females = training['gender'] ==  0
training_no_gender = training['gender'].isnull()
training.loc[training_males, ['gender']] = 'Male'
training.loc[training_no_gender, ['gender']] = 'No Gender'
training.loc[training_females, ['gender']] = 'Female'

training_basestats = training[column_names1].copy()

training_not_basestats = training[column_names2].copy()
training_not_basestats = pd.get_dummies(data=training_not_basestats,
                                         columns=['gender'])

#print(training_not_basestats)

test_males = test['gender'] ==  1
test_females = test['gender'] ==  0
test_no_gender = test['gender'].isnull()
test.loc[test_males, ['gender']] = 'Male'
test.loc[test_no_gender, ['gender']] = 'No Gender'
test.loc[test_females, ['gender']] = 'Female'

test_basestats = test[column_names1].copy()
test_basestats_scaled = test_basestats - FeatExt.pokemon_stats_min
test_basestats_scaled = test_basestats_scaled / FeatExt.pokemon_stats_max

test_not_basestats = test[column_names2].copy()
test_not_basestats = pd.get_dummies(data=test_not_basestats,
                                     columns=['gender'])

###############################################################################
###Data Writing Function
###############################################################################

###data_writer() is a function that simplifies the concatenation and writing
###the csv. Is not restricted to data regarding transformations.
###INPUT: df1 := the non-base stats dataframe
###       df2 := the base stats dataframe
###       csv_name := the desired name for the csv
###       n_components := optional arguement if we only was a certain number of
###                       components from the second datframe
def data_writer(df1, df2, column_names, csv_name, n_components=0):
    
    if n_components > 0:
        df2 = df2.iloc[:, range(0, n_components)]
    data = pd.concat([df1, df2],
                     axis=1,
                     ignore_index=True)
    data.columns = column_names
    csv_name = csv_name + '.csv'
    data.to_csv(csv_name, index=False)
    return(csv_name + ' has been written.')


###test_data_writer() is very similar to data_writer, but it is designed
###for test data where we still need to transform the test data before we write
###the csv.
###INPUT: note_basestats_df := a Pandas dataframe of the non-basestats
###       basestats_scaled_df := a Pandas dataframe of the scaled base stats
###       model := The projection model which does the transformation
###       csv_name := string of the desired name for the csv to have
###       Iso := boolean, and a special arguement if the desired transformation
###              is Isomapping
###OUTPUT: A written csv located in the working directory and a string
###        confirming the data was written.

def test_data_writer(not_basestats_df, basestats_scaled_df, model,
                     csv_name, n_components, Iso=False):
    ###First Pandas-dataframe to be concatenated
    df1 = not_basestats_df
    
    ###Second Pandas-dataframe to be concatenated
    df2 = model.transform(basestats_scaled_df)
    
    if Iso is False:
        df2_column_names = ['PC'+str(i) for i in
                            range(1, (df2.shape[1] + 1))]
    else:
        df2_column_names = [str(i)+'D' for i in
                            range(1, (df2.shape[1] + 1))]
    df2 = pd.DataFrame(data=df2,
                       columns=df2_column_names)
    df2 = df2.iloc[:, range(0, n_components)]
    
    
    ###Pasting the two dataframes together
    column_names = list(training_not_basestats.columns) + df2_column_names[:n_components]
    data = pd.concat([df1, df2],
                     axis=1,
                     ignore_index=True)
    data.columns = column_names
    
    ###Output
    csv_name = csv_name + '.csv'
    data.to_csv(csv_name, index=False)
    return(csv_name + ' has been written.')
        
        

###############################################################################
###Original Data
###############################################################################
    
original_cnames = list(training_not_basestats.columns) + list(training_basestats.columns)
data_writer(training_not_basestats,
            training_basestats,
            original_cnames,
            'original_training')

data_writer(test_not_basestats,
            test_basestats,
            original_cnames,
            'original_test')

###############################################################################
###Total-Aggregation Data
###############################################################################

total_cnames = list(training_not_basestats.columns) + ['total_stats']

training_total = (training_basestats.hp + training_basestats.attack +
                 training_basestats.spattack + training_basestats.defense +
                 training_basestats.spdefense + training_basestats.speed)
data_writer(training_not_basestats,
            training_total,
            total_cnames,
            'total_training')

test_total = (test_basestats.hp + test_basestats.attack +
              test_basestats.spattack + test_basestats.defense +
              test_basestats.spdefense + test_basestats.speed)
data_writer(test_not_basestats,
            test_total,
            total_cnames,
            'total_test')

###############################################################################
###Offensive-Defensive-Aggregation Data
###############################################################################

off_def_cnames = list(training_not_basestats.columns) + ['offensive_stats',
                                                        'defensive stats']

training_offensive = (training_basestats.attack + training_basestats.spattack +
                      training_basestats.speed)
training_defensive = (training_basestats.hp + training_basestats.defense +
                      training_basestats.spdefense)
training_off_def = pd.concat([training_offensive, training_defensive],
                             axis=1,
                             ignore_index=True)
data_writer(training_not_basestats,
            training_off_def,
            off_def_cnames,
            'offensive_defensive_training')

test_offensive = (test_basestats.attack + test_basestats.spattack +
                  test_basestats.speed)
test_defensive = (test_basestats.hp + test_basestats.defense +
                  test_basestats.spdefense)
test_off_def = pd.concat([test_offensive, test_defensive],
                         axis=1,
                         ignore_index=True)
data_writer(test_not_basestats,
            test_off_def,
            off_def_cnames,
            'offensive_defensive_test')

###############################################################################
###PCA Data
###############################################################################

PCA_cnames = list(training_not_basestats.columns) + FeatExt.basestats_pca_column_names[:3]
data_writer(training_not_basestats,
            FeatExt.basestats_PCscores,
            PCA_cnames,
            'PCA_training',
            3)

test_data_writer(test_not_basestats,
                 test_basestats_scaled,
                 FeatExt.basestats_pca,
                 'PCA_test',
                 3)

###############################################################################
###Polynomial-Kernal PCA Data
###############################################################################

poly_KPCA_cnames = list(training_not_basestats.columns) + FeatExt.basestats_kpca_poly_column_names[:4]
data_writer(training_not_basestats,
            FeatExt.basestats_kPCscores_poly,
            poly_KPCA_cnames,
            'poly_KPCA_training',
            4)

test_data_writer(test_not_basestats,
                 test_basestats_scaled,
                 FeatExt.basestats_kpca_poly,
                 'poly_KPCA_test',
                 4)

###############################################################################
###RBF-Kernal PCA Data
###############################################################################

RBF_KPCA_cnames = list(training_not_basestats.columns) + FeatExt.basestats_kpca_RBF_column_names[:30]
data_writer(training_not_basestats,
            FeatExt.basestats_kPCscores_RBF,
            RBF_KPCA_cnames,
            'RBF_KPCA_training',
            30)

test_data_writer(test_not_basestats,
                 test_basestats_scaled,
                 FeatExt.basestats_kpca_RBF,
                 'RBF_KPCA_test',
                 30)

###############################################################################
###Cosine-Kernal PCA Data
###############################################################################

cosine_KPCA_cnames = list(training_not_basestats.columns) + FeatExt.basestats_kpca_cosine_column_names[:5]
data_writer(training_not_basestats,
            FeatExt.basestats_kPCscores_cosine,
            cosine_KPCA_cnames,
            'cosine_KPCA_training',
            5)

test_data_writer(test_not_basestats,
                 test_basestats_scaled,
                 FeatExt.basestats_kpca_cosine,
                 'cosine_KPCA_test',
                 5)

###############################################################################
###Isomap Data
###############################################################################

isomap_cnames = list(training_not_basestats.columns) + ['1D', '2D', '3D']
data_writer(training_not_basestats,
            FeatExt.basestats_isomap_df,
            isomap_cnames,
            'isomap_training',
            3)

test_data_writer(test_not_basestats,
                 test_basestats_scaled,
                 FeatExt.basestats_isomap,
                 'isomap_test',
                 3,
                 Iso=True)
