# -*- coding: utf-8 -*-

"""
model_builder_repeater

Christian Rodriguez
crodriguez0874@gmail.com
08/09/19

Summary - This script simply runs the model_builder() function for the 8 data
sets we created in the Tuning_Prediction_Prep.py script. The resulting models
and associated information are then compiled into a data frame.

"""

###############################################################################
###Loading libraries
###############################################################################
                 
import pandas as pd
import model_builder as mb

###############################################################################
###Training the best model per data set
###############################################################################

original_model = mb.model_builder('original')
off_def_model = mb.model_builder('offensive_defensive')
total_model = mb.model_builder('total')
isomap_model = mb.model_builder('isomap')
PCA_model = mb.model_builder('PCA')
poly_KPCA_model = mb.model_builder('poly_KPCA')
RBF_KPCA_model = mb.model_builder('RBF_KPCA')
cosine_KPCA_model = mb.model_builder('cosine_KPCA')

###############################################################################
###Compiling the results into a data frame
###############################################################################

model_statistics_df = [original_model, off_def_model, total_model, isomap_model, 
                       PCA_model, poly_KPCA_model, RBF_KPCA_model, cosine_KPCA_model]

model_statistics_df = pd.DataFrame(model_statistics_df,
                                   columns=['data_set', 'model',
                                            'test_accuracy', 'parameters'])

model_statistics_df.to_csv('models_statistics.csv', encoding='utf-8',
                           sep=',', index=False)