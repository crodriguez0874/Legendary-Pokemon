# -*- coding: utf-8 -*-

"""
model_builder

Christian Rodriguez
crodriguez0874@gmail.com
08/09/19

Summary - For a data set, the model_builder() function in this script will train
and tune 7 machine learning models via cross validation. We make sure to scale
the data for models sensitive to the scaling of different features. For most of
our tuning processes, we do 2 rounds of random search to optimally tune the
parameters of our models. The first round considers a wide range of values, and
the second round narrows down the range of values considered based on what set
of paramters had the maximal cross validation accuracy. The 7 models trained and
tuned are: lasso logistic regression, ridge logistic regression, elastic logistic
regression, random forest, gradient boosting, k-nearest-neighbor, and support
vector machine. Then, the function will compare the models' accuracies and
output a list with: the data set used, the model with the maximum accuracy on
the test set, the accuracy itself, and the model's function parameters to
reproduce it. In addition, we also calculate the accuracy of the naive
classifier to have a baseline accuracy to compare to. The input should be a
string as one of the following:
    
original

offensive_defensive

total

isomap

PCA

poly_KPCA

RBF_KPCA

cosine_KPCA

The names mentioned above correspond to the different data sets generated in the
feature extraction milestone of this analysis. It is crucial that those data sets
are in a folder called 'data' within the current working directory so that this
function works. Other data sets may be considered; however, if one wishes to do
so, then they should make sure to edit the error handling case in this function
that checks if the input is one of the possible choice and pathway name. Also,
make sure that data set is already one hot encoded, all the features are
quantitative or factor, and that the response is the first column of the data
set.
"""

###############################################################################
###Loading libraries
###############################################################################

import pandas as pd
import scipy.stats as stats
import numpy as np
import sklearn.svm as svm
import sklearn.ensemble as ens
import sklearn.linear_model as lin
import sklearn.neighbors as nei
import sklearn.model_selection as mod
import tuner_refiner as tr

###############################################################################
###Function
###############################################################################

def model_builder(data_set):
    
    ###Error handling when the input is not a string
    if not isinstance(data_set, str):
        
        return('ERROR: Input must be a string.')
    
    ###Error handling when the input is not one of the allowed strings.
    data_sets = ['original', 'offensive_defensive', 'total', 'isomap',
                 'PCA', 'poly_KPCA', 'RBF_KPCA', 'cosine_KPCA']
    
    if data_set not in data_sets:
        
        possible_choices = 'original, offensive_defensive, total, isomap, PCA, poly_KPCA, RBF_KPCA, or cosine_KPCA'
        return('ERROR: Input must be a character string as: ' + possible_choices + ' (or the name of the csv). Also, make sure those data sets are in the working directory.')
    
    ###The core purpose of this script.
    else:
    
        #######################################################################
        ###Initializing the data
        #######################################################################
        
        training_path = 'data/' + data_set + '_training.csv'
        test_path = 'data/' + data_set + '_test.csv'
    
        training = pd.read_csv(training_path, encoding='utf-8')
        test = pd.read_csv(test_path, encoding='utf-8')
        
        #######################################################################
        ###Scaling the data
        #######################################################################
        
        ###Scale the data for models sensitive to different feature scalings
        training_max = training.max()
        training_min = training.min()
        
        training_scaled = training - training_min
        training_scaled = training_scaled / training_max
        
        test_scaled = test - training_min
        test_scaled = test_scaled / training_max
        
        #######################################################################
        ###Create lists for parameters statistics
        #######################################################################
        
        models = []
        models_acc = []
        models_parameters = []
        
        #######################################################################
        ###Niave Classifier
        #######################################################################
        
        ###What if we predicted all the pokemon as non-legendary?
        naive_classifier_accuracy = sum(test.is_legendary == 0)/len(test)
        
        ###Storing the model statistics
        models = models + ['naive classifier']
        models_acc = models_acc + [naive_classifier_accuracy]
        models_parameters = models_parameters + [None]
        
        #######################################################################
        ####Ridge
        #######################################################################
        
        ###Tuning and Fitting
        ridge_classifier = lin.LogisticRegressionCV(cv=10,
                                                    penalty='l2',
                                                    solver='liblinear',
                                                    random_state=1)
        ridge_classifier.fit(training_scaled.iloc[:, range(1,training_scaled.shape[1])],
                             training_scaled.iloc[:, 0])
        
        
        ###Predictions and Accuracy
        ridge_predictions = ridge_classifier.predict(test_scaled.iloc[:, range(1, test_scaled.shape[1])])
        ridge_accuracy = sum(ridge_predictions == test_scaled.iloc[:, 0])/test_scaled.shape[0]
        
        ###Storing the model statistics
        models = models + ['Ridge Logistic Regression']
        models_acc = models_acc + [ridge_accuracy]
        models_parameters = models_parameters + [ridge_classifier.get_params]
        
        #######################################################################
        ###Lasso
        #######################################################################
        
        ###Tuning and Fitting
        lasso_classifier = lin.LogisticRegressionCV(cv=10,
                                                    penalty='l1',
                                                    solver='liblinear',
                                                    random_state=1)
        lasso_classifier.fit(training_scaled.iloc[:, range(1,training_scaled.shape[1])],
                             training_scaled.iloc[:, 0])
        
        
        ###Predictions and Accuracy
        lasso_predictions = lasso_classifier.predict(test_scaled.iloc[:, range(1, test_scaled.shape[1])])
        lasso_accuracy = sum(lasso_predictions == test_scaled.iloc[:, 0])/test_scaled.shape[0]
        
        ###Storing the model statistics
        models = models + ['Lasso Logistic Regression']
        models_acc = models_acc + [lasso_accuracy]
        models_parameters = models_parameters + [lasso_classifier.get_params]
        
        #######################################################################
        ###Elastic Net
        #######################################################################
        
        ###Tuning and Fitting
        elasticnet_classifier = lin.LogisticRegressionCV(cv=10,
                                                        penalty='elasticnet',
                                                        solver='saga',
                                                        l1_ratios=np.arange(.01, .09, .01),
                                                        random_state=1)
        elasticnet_classifier.fit(training_scaled.iloc[:, range(1,training_scaled.shape[1])],
                                  training_scaled.iloc[:, 0])
        
        
        ###Predictions and Accuracy
        elasticnet_predictions = elasticnet_classifier.predict(test_scaled.iloc[:, range(1, test_scaled.shape[1])])
        elasticnet_accuracy = sum(elasticnet_predictions == test_scaled.iloc[:, 0])/test_scaled.shape[0]
        
        ###Storing the model statistics
        models = models + ['Elastic Net Logistic Regression']
        models_acc = models_acc + [elasticnet_accuracy]
        models_parameters = models_parameters + [elasticnet_classifier.get_params]
        
        
        #######################################################################
        ###Random Forest
        #######################################################################
        
        ###Initializing the random search grid
        criterion = ['gini', 'entropy']
        n_estimators = [200*x for x in range(1,11)]
        min_samples_leaf = [1] + [int(round(0.01*len(training), 0)),
                                  int(round(0.05*len(training), 0)),
                                  int(round(0.10*len(training), 0)),
                                  int(round(0.20*len(training), 0))]
        min_samples_split = [2] + [int(round(0.01*len(training), 0)),
                                   int(round(0.05*len(training), 0)),
                                   int(round(0.10*len(training), 0)),
                                   int(round(0.20*len(training), 0))]
        max_depth = [5*x for x in range(1,20)] + [None]
        
        random_grid = {'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'n_estimators': n_estimators,
                       'criterion': criterion}
        
        ###Tuning via random search CV (1st round)
        rf_estimator = ens.RandomForestClassifier(max_features='sqrt',
                                                  bootstrap=True,
                                                  random_state=1)
        
        rf_classifier = mod.RandomizedSearchCV(estimator=rf_estimator,
                                               param_distributions=random_grid,
                                               cv=5,
                                               n_jobs=1,
                                               n_iter=20,
                                               random_state=1)
        
        rf_classifier.fit(training.iloc[:, range(1,training.shape[1])],
                          training.iloc[:, 0])
        
        ###Refining the random search grid
        parameters = list(rf_classifier.best_params_.values())
        
        n_estimators = tr.tuning_refiner(parameters[0], n_estimators, integer=True)
        min_samples_split = tr.tuning_refiner(parameters[1], min_samples_split, integer=True)
        min_samples_leaf = tr.tuning_refiner(parameters[2], min_samples_leaf, integer=True)
        max_depth = tr.tuning_refiner(parameters[3], max_depth, integer=True)
            
        random_grid = {'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'n_estimators': n_estimators}
        
        ###Tuning via random search CV (2nd round)
        rf_estimator = ens.RandomForestClassifier(max_features='sqrt',
                                                  bootstrap=True,
                                                  random_state=1)
        
        rf_classifier = mod.RandomizedSearchCV(estimator=rf_estimator,
                                               param_distributions=random_grid,
                                               cv=5,
                                               n_jobs=1,
                                               n_iter=20,
                                               random_state=1)
        
        rf_classifier.fit(training.iloc[:, range(1,training.shape[1])],
                          training.iloc[:, 0])
        
        ###Predictions and Accuracy
        rf_predictions = rf_classifier.predict(test.iloc[:, range(1, test.shape[1])])
        rf_accuracy = sum(rf_predictions == test.iloc[:, 0])/test.shape[0]
        
        ###Storing the model statistics
        models = models + ['Random Forest']
        models_acc = models_acc + [rf_accuracy]
        models_parameters = models_parameters + [rf_classifier.get_params]
        
        #######################################################################
        ###Gradient Boosting
        #######################################################################
        
        ###Initializing the random search grid
        learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
        n_estimators = [200*x for x in range(1,11)]
        max_depth = [5*x for x in range(1,20)] + [None]
        min_samples_split = [2] + [int(round(0.01*len(training), 0)),
                                   int(round(0.05*len(training), 0)),
                                   int(round(0.10*len(training), 0)),
                                   int(round(0.20*len(training), 0))]
        
        min_samples_leaf = [1] + [int(round(0.01*len(training), 0)),
                                  int(round(0.05*len(training), 0)),
                                  int(round(0.10*len(training), 0)),
                                  int(round(0.20*len(training), 0))]
        #max_features = list(range(2, (training.shape[1] - 1), 2))
        
        random_grid = {'learning_rate': learning_rate,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf}
        
        ###Tuning via random search CV (1st round)
        gbm_estimator = ens.GradientBoostingClassifier(random_state=1)
        
        gbm_classifier = mod.RandomizedSearchCV(estimator=gbm_estimator,
                                                param_distributions=random_grid,
                                                cv=5,
                                                n_jobs=1,
                                                n_iter=20,
                                                random_state=1)
        
        gbm_classifier.fit(training.iloc[:, range(1,training.shape[1])],
                           training.iloc[:, 0])
        
        ###Refining the random search grid
        parameters = list(gbm_classifier.best_params_.values())
        
        n_estimators = tr.tuning_refiner(parameters[0], n_estimators, integer=True)
        min_samples_split = tr.tuning_refiner(parameters[1], min_samples_split, integer=True)
        min_samples_leaf = tr.tuning_refiner(parameters[2], min_samples_leaf, integer=True)
        max_depth = tr.tuning_refiner(parameters[3], max_depth, integer=True)
        learning_rate = tr.tuning_refiner(parameters[4], learning_rate, integer=False)
        
        random_grid = {'learning_rate': learning_rate,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf}
        
        ###Tuning via random search CV (2nd round)
        gbm_estimator = ens.GradientBoostingClassifier(random_state=1)
        
        gbm_classifier = mod.RandomizedSearchCV(estimator=gbm_estimator,
                                                param_distributions=random_grid,
                                                cv=5,
                                                n_jobs=1,
                                                n_iter=20,
                                                random_state=1)
        
        gbm_classifier.fit(training.iloc[:, range(1,training.shape[1])],
                           training.iloc[:, 0])
        
        ###Predictions and Accuracy
        gbm_predictions = gbm_classifier.predict(test.iloc[:, range(1, test.shape[1])])
        gbm_accuracy = sum(gbm_predictions == test.iloc[:, 0])/test.shape[0]
        
        ###Storing the model statistics
        models = models + ['Gradient Boosting Ensemble']
        models_acc = models_acc + [gbm_accuracy]
        models_parameters = models_parameters + [gbm_classifier.get_params]
        
        #######################################################################
        ###K-Nearest-Neighbor
        #######################################################################
        
        ###Initializing the random search grid
        n_neighbors = list(range(3, 33, 3))
        algorithm = ['kd_tree', 'ball_tree']
        leaf_size = [int(round(0.01*len(training), 0)),
                     int(round(0.05*len(training), 0)),
                     int(round(0.10*len(training), 0)),
                     int(round(0.20*len(training), 0))]
        p = [1, 2]
        
        random_grid = {'n_neighbors': n_neighbors,
                       'algorithm': algorithm,
                       'leaf_size': leaf_size,
                       'p': p}
        
        ###Tuning via random search CV (1st round)
        knn_estimator = nei.KNeighborsClassifier()
        
        knn_classifier = mod.RandomizedSearchCV(estimator=knn_estimator,
                                                random_state=1,
                                                param_distributions=random_grid,
                                                cv=5,
                                                n_jobs=1,
                                                n_iter=20)
        
        knn_classifier.fit(training_scaled.iloc[:, range(1,training_scaled.shape[1])],
                           training_scaled.iloc[:, 0])
        
        ###Refining the random search grid
        parameters = list(knn_classifier.best_params_.values())
        
        p = parameters[0]
        
        n_neighbors_LB = parameters[1] - 3
        if n_neighbors_LB < 1:
            n_neighbors_LB = 1
        n_neighbors_UB = parameters[1] + 4
        n_neighbors = range(n_neighbors_LB, n_neighbors_UB)
        
        leaf_size = tr.tuning_refiner(parameters[2], leaf_size)
        
        algorithm = parameters[3]
        
        random_grid = {'n_neighbors': n_neighbors,
                       'leaf_size': leaf_size}
        
        ######Tuning via random search CV (2nd round)
        knn_estimator = nei.KNeighborsClassifier(p=p,
                                                 algorithm=algorithm)
        
        knn_classifier = mod.RandomizedSearchCV(estimator=knn_estimator,
                                                random_state=1,
                                                param_distributions=random_grid,
                                                cv=5,
                                                n_jobs=1,
                                                n_iter=20)
        
        knn_classifier.fit(training_scaled.iloc[:, range(1,training_scaled.shape[1])],
                           training_scaled.iloc[:, 0])
        
        ###Predictions and Accuracy
        knn_predictions = knn_classifier.predict(test_scaled.iloc[:, range(1, test_scaled.shape[1])])
        knn_accuracy = sum(knn_predictions == test_scaled.iloc[:, 0])/test_scaled.shape[0]
        
        ###Storing the model statistics
        models = models + ['K-Nearest-Neighbor']
        models_acc = models_acc + [knn_accuracy]
        models_parameters = models_parameters + [knn_classifier.get_params]
        
        #######################################################################
        ###Support Vector Machine
        #######################################################################
        
        ###Initializing the random search grid
        kernel = ['linear', 'rbf', 'poly']
        C = stats.uniform(1, 50)
        gamma = stats.uniform(0.001, 0.500)
        degree = [2, 3, 4, 5]
        
        random_grid = {'kernel': kernel,
                       'C': C,
                       'gamma': gamma,
                       'degree': degree}
        
        ###Tuning via random search CV (1st round)
        svm_estimator = svm.SVC()
        svm_classifier = mod.RandomizedSearchCV(estimator=svm_estimator,
                                                param_distributions=random_grid,
                                                random_state=1,
                                                cv=5,
                                                n_jobs=1,
                                                n_iter=20)
        
        svm_classifier.fit(training_scaled.iloc[:, range(1,training_scaled.shape[1])],
                           training_scaled.iloc[:, 0])
        
        ###Refining parameters
        parameters = list(svm_classifier.best_params_.values())
        
        C = stats.uniform(0.80*parameters[0], 1.2*parameters[0])
        degree = parameters[1]
        gamma = stats.uniform(0.80*parameters[2], 1.2*parameters[2])
        kernel = parameters[3]
        
        random_grid = {'C': C,
                       'gamma': gamma}
        
        ###Tuning via random search CV (2nd round)
        svm_estimator = svm.SVC(kernel=kernel,
                                degree=degree)
        svm_classifier = mod.RandomizedSearchCV(estimator=svm_estimator,
                                                param_distributions=random_grid,
                                                random_state=1,
                                                cv=5,
                                                n_jobs=1,
                                                n_iter=20)
        
        svm_classifier.fit(training_scaled.iloc[:, range(1,training_scaled.shape[1])],
                           training_scaled.iloc[:, 0])
        
        ###Predictions and Accuracy
        svm_predictions = svm_classifier.predict(test_scaled.iloc[:, range(1, test_scaled.shape[1])])
        svm_accuracy = sum(svm_predictions == test_scaled.iloc[:, 0])/test_scaled.shape[0]
        
        ###Storing the model statistics
        models = models + ['Support Vector Machine']
        models_acc = models_acc + [svm_accuracy]
        models_parameters = models_parameters + [svm_classifier.get_params]
        
        #######################################################################
        ###Output
        #######################################################################
        
        ###Output the model with the highest training set accuracy
        optimal_ID = models_acc.index(max(models_acc))
        
        optimal_model = [data_set,
                         models[optimal_ID],
                         models_acc[optimal_ID],
                         models_parameters[optimal_ID]]
        
        return(optimal_model)
