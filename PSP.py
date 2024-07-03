import sys
import os
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, norm
import random
import scipy as sc
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold,GridSearchCV
from itertools import repeat
from yellowbrick.regressor import AlphaSelection
from os import listdir
from os.path import isfile, join
from Bio.SeqUtils import seq1
from Bio.PDB import *
import Bio
from sklearn.decomposition import PCA
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

############################################################################################################################################
# Lasso
############################################################################################################################################

def normalize_features(train_x, features_cols):
    """
    Normalize the training set using MinMaxScaler.

    Parameters:
    train_x (pd.DataFrame): The training data.
    features_cols (list): List of feature column names.

    Returns:
    pd.DataFrame: Normalized training data.
    """
    # normalize training set
    scaler_train = MinMaxScaler()
    train_x_transformed = scaler_train.fit_transform(train_x)
    train_x_transformed = pd.DataFrame(train_x_transformed, columns=features_cols)

    return train_x_transformed


def LASSO_model(DB, features, target, alphas):
    """
    Trains Lasso regression models for each unique protein in the dataset.

    Parameters:
    DB (pd.DataFrame): DataFrame containing the dataset.
    features (list): List of feature names.
    target (str): Name of the target variable.
    alphas (list): List of alpha values to search over for Lasso.

    Returns:
    pd.DataFrame: DataFrame with best alpha and coefficients for each protein.
    """
    collect = {}
    for prot in DB.protein.unique():
        db_protein = DB[DB.protein==prot].reset_index(drop=True)
        X,y  = db_protein[features], db_protein[target]
        X_scaled = normalize_features(X, features)
        model = Lasso(fit_intercept = True, random_state=123,tol=1e-3)

        # define grid
        grid = dict(alpha=alphas)
        #cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=123)
        # define search
        search = GridSearchCV(model, grid, cv=10, scoring='neg_mean_absolute_error')
        results = search.fit(X_scaled, y)
        best = results.best_estimator_
        best.fit(X_scaled,y)
        print(prot, '      best alpha:', '{:.6f}'.format(best.alpha))
        collect[prot] = [best.alpha, list(best.coef_)]
    collect_df = feature_plot_prep_lasso(DB,collect, features)
    return collect_df

def feature_plot_prep_lasso(DB, collect, features_list):
    """
    Prepares a DataFrame for feature plotting from the Lasso model results.

    Parameters:
    DB (pd.DataFrame): DataFrame containing the dataset.
    collect (dict): Dictionary with best alpha and coefficients for each protein.
    features_list (list): List of feature names.

    Returns:
    pd.DataFrame: DataFrame with proteins, alphas, features, and their corresponding values.
    """
    mykeys, myvals, myalphas =[],[],[]

    for k in collect.keys():
        mykeys.extend(repeat(k,len(features_list)))

    for val in collect.values():
        myvals.extend(val[1])
        myalphas.extend(repeat(val[0], len(features_list)))

    collect_df = pd.DataFrame({
        'proteins': mykeys,
        'alphas': myalphas,
        'features': features_list * len(DB.protein.unique()),
        'values': myvals})

    return collect_df


############################################################################################################################################
# Protein-Specific Predictor
############################################################################################################################################

def MLR_LOPO_protein(db_protein, features, target, predictor_name, undersample, path_log):

    """
    Performs Leave-One-position-Out (LOPO) cross-validation using Multiple Linear Regression (MLR).

    Parameters:
    db_protein (pd.DataFrame): DataFrame containing the protein data.
    features (list): List of feature column names.
    target (str): Name of the target variable.
    predictor_name (str): Name of the predictor column to be added to the test DataFrame.
    undersample (bool): Whether to apply undersampling.
    path_log (str): Path to save the debug log file.

    Returns:
    tuple: Tuple containing:
        - pd.DataFrame: DataFrame with predicted values.
        - pd.DataFrame: DataFrame with statistical results.
        - dict: Dictionary with positions and their corresponding model coefficients (values) for each protein (key).
                Format: {protein: [[position, array of coefficients], ...]}
    """


    collect_coefs = []
    if undersample==True:
        print('Model with undersampling...')
    elif undersample == False:
        print('Model without undersampling...')
    else:
        sys.exit('undersample parameter not defined.')

    protein = db_protein.protein.unique()[0]
    db_protein_predicted = pd.DataFrame()

    debug = pd.DataFrame(columns=['protein','test_position/variant','GMM_threshold', 'before1', 'before2', 'after1', 'after2', 'train_size','test_size','predicted_accumulated_size'])
    stats = pd.DataFrame(columns = ['protein', 'model', 'pearson','spearman'])

    for position in db_protein.pos.unique():
        train_db = db_protein[~db_protein.pos.isin([position])].reset_index(drop=True).copy()
        test_db = db_protein[db_protein.pos.isin([position])].reset_index(drop=True).copy()
        train_x_scaled, train_y, test_x_scaled, threshold_prot, counting = train_test_prepare(train_db, test_db, features, target,undersample)

        # train the model & predict
        lm = LinearRegression(fit_intercept=True).fit(train_x_scaled, train_y)
        collect_coefs.append([position,lm.coef_])

        test_db[predictor_name] = [round(a,3) for a in lm.predict(test_x_scaled)]
        db_protein_predicted = pd.concat([db_protein_predicted, test_db])

        debug.loc[len(debug)] = [protein, position, threshold_prot, counting[0], counting[1], counting[2], counting[3],
                                 len(train_x_scaled), len(test_db), str(len(db_protein_predicted))+'/'+str(len(db_protein))]


    debug.to_csv(path_log+protein.replace(' ','_')+'_PSP_LOPO_undersampling.csv',sep='\t')
    # calculate stats
    r, rho = calculate_stats(db_protein_predicted, target, predictor_name)

    if len(db_protein) != len(db_protein_predicted):
        sys.exit(f'Before/after sizes do not match.')



    stats.loc[len(stats)] = [protein, predictor_name, r, rho]
    db_protein_predicted = db_protein_predicted.reset_index(drop=True)
    return db_protein_predicted, stats, collect_coefs


############################################################################################################################################

def train_test_prepare(train_db, test_db, features, target, undersample):

    # check train/test are splitted correctly
    if len(train_db) < len(test_db):
        sys.exit('ERROR. Training set is smaller than testing.')
    if len(test_db) > 1 and len(test_db.pos.unique()) !=1:
        sys.exit('ERROR. Test set has more than one position.')

    if undersample == True:
        # get the threshold
        threshold_prot, _ = GMM_interaction(train_db)

        # undersample the training set
        train_db_undersampled, counting = Undersample_Assay(train_db, threshold_prot)

        # train sets
        train_x, train_y = train_db_undersampled[features], train_db_undersampled[target]

    elif undersample == False:
        threshold_prot, _ = GMM_interaction(train_db)
        # train sets
        train_x, train_y = train_db[features], train_db[target]
        counting = ['-','-','-','-']

    # test set
    test_x = test_db[features]
    # feature scaling
    train_x_scaled, test_x_scaled = normalize_features_train_test(train_x, test_x, features)

    return train_x_scaled, train_y, test_x_scaled, threshold_prot, counting

############################################################################################################################################
def normalize_features_train_test(train_x, test_x, features_cols):
    """
    Normalizes the training and testing datasets using MinMaxScaler.
    Returns:
        - pd.DataFrame: Normalized training dataset.
        - pd.DataFrame: Normalized testing dataset.
    """
    # Normalize training set
    scaler_train = MinMaxScaler()
    train_x_transformed = scaler_train.fit_transform(train_x)
    train_x_transformed = pd.DataFrame(train_x_transformed, columns=features_cols)

    # Normalize testing set using the scaler fitted on the training set
    test_x_transformed = scaler_train.transform(test_x)
    test_x_transformed = pd.DataFrame(test_x_transformed, columns=features_cols)

    return train_x_transformed, test_x_transformed

############################################################################################################################################

def GMM_interaction(db_prot, plot=False):
    '''
    Building gaussian mixture models with two components for continuous data

    Returns:
    - threshold: (float) the intersection point that separates two peaks
    - prot_files: by default returns empty list.
                  if plot = True, returns a list of values required for making GMM plots.
                  -- gmm: Gaussian Mixture Model generated using data given
                  -- X: continuous data points as numpy array
                  -- x: numpy array for the x axis -calculated using min&max values of X
                  -- y1, y2: gaussian curves
                  -- w1, w2: weights of the components
                  -- idxs: list of indices of the intersection points on x axis
    '''
    prot = db_prot.protein.unique()[0]
    X = db_prot['score_log_normalized'].to_numpy().reshape(-1,1)
    x = np.linspace(min(X)[0]-0.1,max(X)[0]+0.1,1000)

    gmm = GaussianMixture(n_components=2, random_state=23).fit(X)
    m1, m2 = gmm.means_
    w1, w2 = gmm.weights_
    c1, c2 = gmm.covariances_
    std1 = np.sqrt(c1[0][0])
    std2 = np.sqrt(c2[0][0])

    y1=sc.stats.norm.pdf(x,m1[0],std1)
    y2=sc.stats.norm.pdf(x,m2[0],std2)

    # indices of the intersection points
    idxs=np.argwhere(np.diff(np.sign(y1*w1 - y2*w2))).flatten()

    # for four proteins, the correct intersection point is the one with index 1, so make sure the to select for that one
    threshold = round(x[idxs[1]],2) if len(idxs)>1 and prot in ['MAPK1','SRC','TPK1','UBE2I'] else round(x[idxs[0]],2)

    #print('Threshold for protein', prot, ':', threshold)

    plot_files = [gmm, X,x,y1,y2, w1,w2,idxs] if plot == True else []

    return threshold, plot_files

############################################################################################################################################


def Undersample_Assay(db_org, threshold):
    """
    Performs undersampling on a dataset based on a given threshold.

    Parameters:
    db_org (pd.DataFrame): Original dataset.
    threshold (float): Threshold for categorizing the score_log_normalized values.

    Returns:
    tuple: Tuple containing:
        - pd.DataFrame: Undersampled dataset.
        - list: List of group distributions before and after undersampling.
    """
    db = db_org.copy()
    db['group'] = np.NaN
    db.loc[db.score_log_normalized < threshold, 'group'] = 'Below thr.'
    db.loc[db.score_log_normalized >= threshold, 'group'] = 'Above thr.'

    x, y = db[db_org.columns], db['group']
    #print('Threshold', threshold,'\n\nOriginal distribution:', Counter(y))
    undersample = RandomUnderSampler(sampling_strategy='majority', random_state=1234)
    x_over, y_over = undersample.fit_resample(x, y)
    #print('\nAfter undersampling  :',Counter(y_over))
    db_undersampled= x_over.copy()

    # Check if undersampling was successful
    if len(db_undersampled) >= len(db):
        sys.exit('undersampling failed.')

    # Calculate group distributions before and after undersampling
    group1_before, group2_before = str(Counter(y))[10:-2].split(',')[0], str(Counter(y))[10:-2].split(',')[1]
    group1_after, group2_after = str(Counter(y_over))[10:-2].split(',')[0], str(Counter(y_over))[10:-2].split(',')[1]

    return db_undersampled, [group1_before, group2_before, group1_after, group2_after]


############################################################################################################################################


def calculate_stats(db_protein_predicted, target, predictor_name):
    """
    Calculates Pearson and Spearman correlation coefficients between the target and predictor.

    Parameters:
    db_protein_predicted (pd.DataFrame): DataFrame containing the predicted values.
    target (str): Name of the target variable.
    predictor_name (str): Name of the predictor variable.

    Returns:
    tuple: Tuple containing:
        - float: Pearson correlation coefficient rounded to 2 decimal places.
        - float: Spearman correlation coefficient rounded to 2 decimal places.
    """
    db_protein_predicted = db_protein_predicted.reset_index(drop=True)

    r, p = pearsonr(db_protein_predicted[target],db_protein_predicted[predictor_name])
    rho, p = spearmanr(db_protein_predicted[target],db_protein_predicted[predictor_name])
    print(f'{db_protein_predicted.protein.unique()[0]}\npearson: {round(r,2)}, spearman: {round(rho,2)}\n__________________________\n\n')

    # Return rounded correlation coefficients
    return round(r,2), round(rho,2)

############################################################################################################################################
# Cross Prediction
############################################################################################################################################

def train_test_prepare_cross(train_db, test_db, features, target, undersample):

    """
    Prepares training and testing datasets, with an option to undersample the training set.
    """

    if undersample == True:
        # get the threshold
        threshold_prot, _ = GMM_interaction(train_db)

        # undersample the training set
        train_db_undersampled, counting = Undersample_Assay(train_db, threshold_prot)
        # train sets
        train_x, train_y = train_db_undersampled[features], train_db_undersampled[target]

    elif undersample == False:
        # train sets
        train_x, train_y = train_db[features], test_db[target]
        counting = ['-','-','-','-']
    threshold_prot, _ = GMM_interaction(train_db)
    # test set
    test_x = test_db[features]
    # feature scaling
    train_x_scaled, test_x_scaled = normalize_features_train_test(train_x, test_x, features)

    return train_x_scaled, train_y, test_x_scaled, threshold_prot, counting


############################################################################################################################################

def Cross_Predictions_MLR(DB, prot_list, path_save, path_log, predictor_name, features, target, undersample):
    """
    Perform cross-predictions using Multiple Linear Regression (MLR) across different proteins.

    Parameters:
    - DB (DataFrame): The dataset containing protein data.
    - prot_list (list): List of proteins to be used for cross-prediction.
    - path_save (str): Path to save the prediction results.
    - path_log (str): Path to save the debug logs.
    - predictor_name (str): Name of the predictor column.
    - features (list): List of feature column names to be used.
    - target (str): The target column name.
    - undersample (bool): Flag indicating whether to undersample the training set.

    Returns:
    - cross_predictions (DataFrame): DataFrame containing cross-prediction statistics.
    """

    cp_list = []
    debug = pd.DataFrame(columns=['protein_train','GMM_threshold','protein_test', 'train_before1', 'train_before2', 'train_after1', 'train_after2', 'train_size','test_size'])
    stats = pd.DataFrame(columns = ['protein', 'model', 'pearson','spearman'])

    for prot_train in prot_list:


        db_protein_train = DB[DB.protein==prot_train].reset_index(drop=True).copy()
        print('==============================================================')
        print('       PROTEIN training:', prot_train, len(db_protein_train))
        print('==============================================================')

        topla = pd.DataFrame()

        prot_test_list =[p for p in prot_list if p != prot_train]
        for prot_test in prot_test_list:
            db_protein_test = DB[DB.protein==prot_test].reset_index(drop=True).copy()

            print(f'testing... {prot_test}')
            train_x_scaled, train_y, test_x_scaled, threshold_prot, counting = train_test_prepare_cross(db_protein_train, db_protein_test, features, target,undersample)
            if len(db_protein_test) != len(test_x_scaled):
                sys.exit('protein test length different than scaled test x.')

            # train the model & predict
            lm = LinearRegression(fit_intercept = True).fit(train_x_scaled, train_y)
            db_protein_test[predictor_name] = lm.predict(test_x_scaled)
            db_protein_test[predictor_name] = round(db_protein_test[predictor_name],3)

            debug.loc[len(debug)] = [prot_train, threshold_prot, prot_test, counting[0], counting[1], counting[2], counting[3],
                                 len(train_x_scaled), len(db_protein_test)]

            # calculate stats
            r, rho = calculate_stats(db_protein_test, target, predictor_name)

            cp_list.append([predictor_name, prot_train, prot_test, r, rho])

            # save predictions of each trained predictor
            db_protein_test['tested_protein'] = prot_test
            topla = pd.concat([topla,db_protein_test[['tested_protein','variant',target,predictor_name]]])

        topla['trained_protein'] = prot_train
        topla.to_csv(f'{path_save}train_{prot_train}_predict_rest.csv')

    # save stats in a table
    cross_predictions = pd.DataFrame(cp_list, columns=['model', 'trained_protein','tested_protein','pearson','spearman'])

    debug.to_csv(path_log + 'CROSS_debug_'+predictor_name +'.csv', index=0)

    return cross_predictions

############################################################################################################################################

def tested_protein_csvs(DB, proteins_list, target,predictor_name, path_save_trained, path_save_tested):
    """
    Collect the real prediction values for each trained_protein/predicted_protein combination and save to CSV files.

    Parameters:
    - DB (DataFrame): The dataset containing protein data.
    - proteins_list (list): List of proteins to be used for testing.
    - target (str): The target column name.
    - predictor_name (str): Name of the predictor column.
    - path_save_trained (str): Path where trained model predictions are saved.
    - path_save_tested (str): Path where the collected predictions will be saved.

    Returns:
    - None: The function saves the output CSV files to the specified path.
    """

    for test_protein in proteins_list:
        # Select the test protein data
        tested = DB[DB.protein==test_protein][['protein','variant',target]]
        trained_prots = [t for t in proteins_list if t != test_protein]

        print(f'test protein: {test_protein} // collect preds. from:{len(trained_prots),trained_prots}\n')

        for train in trained_prots:
            # Read the trained protein predictions
            pp = pd.read_csv(f'{path_save_trained}train_{train}_predict_rest.csv',index_col =0)

            # Filter predictions for the current test protein
            train_preds = pp[pp.tested_protein==test_protein]
            # Merge predictions with the test data
            tested = tested.merge(train_preds[['variant',target,predictor_name]], on=['variant', target], how='left')
            tested.rename(columns={predictor_name:predictor_name+ '_trainedby_'+train}, inplace=True)

        # Rename the target column for clarity
        tested.rename(columns={target:target+'_'+test_protein},inplace=True)

        tested.to_csv(path_save_tested + test_protein + '.csv', index=0)

############################################################################################################################################

def function_cross_preds_median(path_save_tested, stats_all, predictor_name, target, proteins_to_be_tested,howmany):

    """
    Perform cross-predictions using the median of the best predictors for each tested protein.

    Parameters:
    - path_save_tested (str): Path where the tested predictions are saved.
    - stats_all (DataFrame): DataFrame containing statistics of all cross-predictions.
    - predictor_name (str): Name of the predictor column.
    - target (str): The target column name.
    - proteins_to_be_tested (list): List of proteins to be tested.
    - howmany (int): Number of top predictors to use for the median prediction.

    Returns:
    - stat_table (DataFrame): DataFrame containing statistics of the median model predictions.
    """

    print(f'{predictor_name}....\n\n')

    stat_table = pd.DataFrame(columns=['tested_protein', 'MedianModel','pearson','spearman','median_of_which_proteins'])

    for tested_protein in proteins_to_be_tested:
        # ROUND 1: SELECTING THE BEST PROTEINS FOR THE MEDIAN
        print(f'___________________________________________________________________________\n:::::::ROUND 1::::::: \n we are going to find who are the best {howmany} predictors for predicting {tested_protein}\n___________________________________________________________________________\n\n')
        proteins_rest = [p for p in proteins_to_be_tested if p != tested_protein]
        p_col,median_col = [],[]
        for p in proteins_rest:
            trained = p
            tested = [rest for rest in proteins_rest if rest != p]
            print(f'TRAIN: {trained}\nTESTED:{tested}')
            collect_pearsons = []
            for t in tested:
                pearson_t = stats_all.loc[(stats_all.trained_protein == p) & (stats_all.tested_protein==t)]['pearson'].values[0]
                #print(f'\t{t}: {pearson_t}')
                collect_pearsons.append(pearson_t)
            median_pearsons = np.median(collect_pearsons)
            print(f'\nMedian of pearsons: {median_pearsons}')
            print('_____________\n')
            p_col.append(p)
            median_col.append(median_pearsons)
        db = pd.DataFrame({'proteins':p_col, 'median_pearsons':median_col})
        db = db.sort_values(by='median_pearsons', ascending=False)
        top_proteins = db.head(howmany).proteins.values
        print(f'best predictors for {tested_protein} are: {top_proteins}\n')
        print(f'___________________________________________________________________________\n:::::::ROUND 2::::::: \n we are going to predict {tested_protein} using median of {top_proteins}\n___________________________________________________________________________\n\n')

        stat_table_perprot = cross_preds_median_one(tested_protein, path_save_tested, top_proteins, predictor_name, target)
        stat_table_perprot['median_of_which_proteins'] = str(top_proteins)


        stat_table = pd.concat([stat_table, stat_table_perprot])

        print('- - - - - - - - - - - - - - - - - - - - - - - \n- - - - - - - - - - - - - - - - - - - - - - -\n\n')
    return stat_table.reset_index(drop=True)

############################################################################################################################################

def cross_preds_median_one(tested_protein, path_save_tested, median_prots_list, predictor_name, target):

    """
    tested_protein = this is the protein you would like to predict
    median_prots_list = list of proteins you want to get predictions from, and to take their predictions' median
    """

    print(f'Tested protein:\t{tested_protein}\n\nProteins selected for median:\n\n  {sorted(median_prots_list)}\n\n')
    cp_list = []

    tested = pd.read_csv(path_save_tested + tested_protein + '.csv')

    pred = predictor_name.split("_")[0]

    newname = f'QAFI({pred}_median_'+str(len(median_prots_list))+')'

    names = [predictor_name + '_trainedby_'+ prot for prot in median_prots_list]
    tested[newname] = tested[names].median(axis=1)
    tested[newname] = round(tested[newname],3 )
    tested['total_medians'+str(len(median_prots_list))] = str(median_prots_list)

    median_pred_name = newname
    target_name = target + '_' + str(tested_protein)

    r, p = pearsonr(tested[target_name],tested[median_pred_name])
    rho, p = spearmanr(tested[target_name],tested[median_pred_name])

    r, rho = round(r,2), round(rho,2)
    print(f'output column name: // {newname} // \n\nmedian of {len(median_prots_list)} predictions: r = {r}, rho = {rho}')

    tested.to_csv(f'{path_save_tested}{tested_protein}_median{len(median_prots_list)}.csv', index=0)

    cp_list.append([tested_protein, median_pred_name, r, rho])
    stat_table = pd.DataFrame(cp_list, columns=['tested_protein', 'MedianModel','pearson','spearman'])

    print('_________________________________________________________________\n')

    return stat_table


############################################################################################################################################

def count_protein_occurrences(input_array):
    """
    Creates a table with the count of each protein name in the input array.

    Parameters:
    - input_array (numpy array): Array of strings, each containing a list of protein names.

    Returns:
    - DataFrame: A table with two columns: 'protein' and 'count', showing the number of occurrences of each protein.
    """
    # Initialize an empty list to store all protein names
    all_proteins = []

    # Loop through each string in the input array
    for proteins_str in input_array:
        # Remove the brackets and split by spaces to get individual protein names
        proteins_list = proteins_str.strip("[]").replace("'", "").split()
        # Extend the all_proteins list with the proteins from the current list
        all_proteins.extend(proteins_list)

    # Create a DataFrame with the count of each protein
    protein_counts = pd.Series(all_proteins).value_counts().reset_index()
    protein_counts.columns = ['protein', 'count']

    return protein_counts
