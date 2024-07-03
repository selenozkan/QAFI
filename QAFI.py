import pandas as pd
import numpy as np
import sys
from Bio.SeqUtils import seq1
from os import listdir
from os.path import isfile, join
import os
import math
from collections import Counter
from sklearn.linear_model import LinearRegression
import PSP



def QAFI_train_psp_test_protein(proteins_10, DB_train, proteins_to_be_tested_uni, output_path, features, target, predictor_name, path_save, path_log):
    """
    Perform cross-predictions using Multiple Linear Regression (MLR) across different proteins.
    NOTE: essentially the same function in Cross_Predictions_MLR in PSP.py with slight modifications
    """
    debug = pd.DataFrame(columns=['protein_train','GMM_threshold','protein_test', 'train_before1', 'train_before2', 'train_after1', 'train_after2', 'train_size','test_size'])

    db_protein_test = pd.read_csv(f'data/proteins/{proteins_to_be_tested_uni}/{proteins_to_be_tested_uni}_featuresAll.csv')

    # structural features (7/9)
    columns_to_update = ['colasi','fraction cons. 3D neighbor','fanc','fbnc','M.J. potential','access.dependent vol.','laar']

    # Update structural features where 'pLDDT bin' is 0
    db_protein_test.loc[db_protein_test['pLDDT bin'] == 0, columns_to_update] = 0.0

    for prot_train in proteins_10:
        db_protein_train = DB_train[DB_train.protein == prot_train].reset_index(drop=True).copy()
        print('==============================================================')
        print('       PROTEIN training:', prot_train, len(db_protein_train))
        print('==============================================================')


        prot_test = db_protein_test.protein.unique()[0]
        uni_test = db_protein_test.uniprot.unique()[0]

        print(f'testing... {prot_test}\n')

        train_x_scaled, train_y, test_x_scaled, threshold_prot, counting = PSP.train_test_prepare_cross(db_protein_train, db_protein_test, features, target, undersample=True)
        if len(db_protein_test) != len(test_x_scaled):
            sys.exit('protein test length different than scaled test x.')

        # train the model & predict
        lm = LinearRegression(fit_intercept=True).fit(train_x_scaled, train_y)
        db_protein_test[predictor_name] = lm.predict(test_x_scaled)
        db_protein_test[predictor_name] = round(db_protein_test[predictor_name], 3)

        debug.loc[len(debug)] = [prot_train, threshold_prot, prot_test, counting[0], counting[1], counting[2], counting[3], len(train_x_scaled), len(db_protein_test)]

        # save predictions of each trained predictor
        db_protein_test['tested_protein'] = prot_test

        predictions = db_protein_test[['tested_protein', 'variant', predictor_name]].copy()
        predictions['trained_protein'] = prot_train
        predictions['tested_uniprot'] = uni_test

        predictions.to_csv(f'{path_save}/{prot_train}/train_{prot_train}_predict_{uni_test}.csv')

    debug.to_csv(path_log + 'CROSS_debug_' + predictor_name + '.csv')


################################################################################################################################################################################

def aggregate_predictions(proteins_10, path_psp, test_protein_uniprot, db_protein_test, predictor_name, output_path):
    """
    Aggregate predictions from 10 PSPs and save the combined results.

    Parameters:
    - proteins_10: List of training protein names.
    - path_psp: Base path to the PSP predictions.
    - test_protein_uniprot: Uniprot ID of the test protein.
    - db_protein_test: DataFrame of the test protein data.
    - predictor_name: Name of the predictor column.
    - output_path: Path to save the aggregated predictions.
    """

    for train in proteins_10:
        # Read the predictions for each training protein
        pp = pd.read_csv(f'{path_psp}/{train}/train_{train}_predict_{test_protein_uniprot}.csv', index_col=0)

        # Merge the predictions with the test protein DataFrame
        db_protein_test = db_protein_test.merge(pp[['variant', predictor_name]], on=['variant'], how='left')

        # Rename the predictor column to include the training protein name
        db_protein_test.rename(columns={predictor_name: predictor_name + '_trainedby_' + train}, inplace=True)

    # Ensure the output directory exists
    os.makedirs(f'{output_path}/{test_protein_uniprot}', exist_ok=True)

    # Save the combined predictions to a CSV file
    db_protein_test.to_csv(f'{output_path}/{test_protein_uniprot}/{test_protein_uniprot}_predictions_all.csv', index=False)



################################################################################################################################################################################

def calculate_median_predictions(db_protein_test, proteins_10, predictor_name):
    """
    Calculate median predictions from multiple trained models for each test protein.

    Parameters:
    - db_protein_test: tested data.
    - proteins_10: List of proteins used for calculating the median.
    - predictor_name: Name of the predictor column.
    """

    tested_protein = db_protein_test.protein.unique()[0]
    print(f'Tested protein:\t{tested_protein}\n\nProteins selected for median:\n\n  {sorted(proteins_10)}\n\n')


    newname = f'QAFI(MLR_median_'+str(len(proteins_10))+')'

    names = [predictor_name + '_trainedby_' + prot for prot in proteins_10]

    print(len(names))

    db_protein_test[newname] = db_protein_test[names].median(axis=1)
    db_protein_test[newname] = round(db_protein_test[newname], 3)
    #db_protein_test['total_medians' + str(len(proteins_10))] = str(proteins_10)

    median_pred_name = newname

    return db_protein_test
