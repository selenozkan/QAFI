import os
import pandas as pd
import numpy as np
from Bio.PDB import *
import Bio
from os import listdir
from os.path import isfile, join
import io
import json
import collections
import sys
from Bio import SeqIO
from math import log
from Bio.SeqUtils import seq1
from scipy.spatial import distance
import shutil


##########################################################################

# Blosum62, PSSM, Shannon's Entropy

##########################################################################

def parse_arff(uniprot):
    # this function is to parse the output of the PatMut pipeline
    # Please check https://github.com/NataliaSirera/patmut to follow the details of how the .arff files are obtained

    # if you change the parameters in the .config file, the first row with variant and number of columns will change
    # in this case, please modify this function for the "skiprows = ..." and the data.columns = ...

    results = 'data/proteins/'+ uniprot + '/'
    path = results + str(uniprot)
    data = pd.read_csv(path+'.arff', skiprows=15,sep=',',header=None)
    data.columns=['variant','uniprot','vdwVolume','hydrophobicity','substitutionMatrix','pssm-native','pssm-mutated','entropy','freqGaps','impRes','tag']
    data = data[['uniprot','variant','substitutionMatrix','pssm-native','entropy','vdwVolume','hydrophobicity']]
    data["first"] = data['variant'].str[0]
    data["pos"] = data['variant'].str[1:-1].astype(np.int64)
    data["second"] = data['variant'].str[-1]
    data = data.sort_values("pos", ascending=True).reset_index(drop=True)
    data = data[['uniprot','variant','first','pos', 'second','substitutionMatrix','pssm-native','entropy']]

    data.rename(columns={'substitutionMatrix':'Blosum62',
                         'pssm-native':'PSSM',
                         'entropy':"Shannon's entropy"}, inplace=True)


    data.to_csv(results + uniprot +'.csv', index=0)

    # additionally, save entropy for AlphaFold features
    data_ent = data[['uniprot','variant',"Shannon's entropy"]].copy()
    data_ent['wt_pos'] = data_ent['variant'].str[:-1]
    data_ent.to_csv(results + uniprot + '_entropy.csv', index=0)

    print(f'outputs saved for {uniprot}:\n\n - {results}{uniprot}.csv\n - {results}{uniprot}_entropy.csv')
    return None

##########################################################################

# Shannon's entropy of seq. neighbours

##########################################################################

def add_entropy_window(uniprot,db_protein):
    '''
    Calculating average of the -3 and +3 neighbour positions' entropies for a given residue
    Returns the protein dataframe with one new column added
    '''

    results = 'data/proteins/'+ uniprot + '/'
    df_entropy = pd.read_csv(results + uniprot + '_entropy.csv').copy()

    df_entropy['pos'] = df_entropy['wt_pos'].str[1:].astype('int')
    df_entropy_unique = df_entropy.drop_duplicates(subset=['wt_pos'], keep="first").reset_index(drop=True)
    positions = df_entropy_unique.pos.unique()

    entropy_window = {}
    for index in range(len(positions)):
        pos = positions[index]
        #print('\n      POSITION',pos)
        left_window_pos,right_window_pos,left_window,right_window = [],[], [], []
        if index < 3: # for cases where less then 3 residue on the left available
            left_window_pos = positions[:index]
            right_window_pos = positions[index+1:index+4]
        elif len(positions) - index <= 3: # for cases where less then 3 residue on the right available
            left_window_pos = positions[index-3:index]
            right_window_pos = positions[index+1:]
        else: # for cases where 3 residues from left and 3 from right available
            left_window_pos = positions[index-3:index]
            right_window_pos = positions[index+1:index+4]

        left_window = list(df_entropy_unique[df_entropy_unique.pos.isin(left_window_pos)]["Shannon's entropy"].values)
        right_window = list(df_entropy_unique[df_entropy_unique.pos.isin(right_window_pos)]["Shannon's entropy"].values)
        #print('left',left_window_pos, left_window,'\nright',right_window_pos, right_window)
        #print('sum of neighbours:', sum(left_window) + sum(right_window), '\nnumber of neighbours:', (len(left_window) + len(right_window)))
        #print('Average entropy window for position', pos, ': ', round((sum(left_window) + sum(right_window)) / (len(left_window) + len(right_window)),3),'\n')

        entWind = (sum(left_window) + sum(right_window)) / (len(left_window) + len(right_window))
        entropy_window[pos] = round(entWind, 3)


    for k,v in entropy_window.items():
        db_protein.loc[db_protein.pos == k, "Shannon's entropy of seq. neighbours"] = v

    return db_protein


##########################################################################

# Neco

##########################################################################

def add_neco(uniprot, db_protein):
    # please refer to our QAFI paper to follow how feature_neco.csv data was calculated
    # Credit for calculation of this feaure: Natàlia Padilla Sirera https://github.com/NataliaSirera
    newcolname = 'neco'
    feature_db = pd.read_csv('data/feature_neco.csv')
    db = db_protein.copy()
    db[newcolname] = np.NaN
    variant_list = db.variant.values
    for variant in variant_list:
        wt_fa = variant[0]
        mut_fa = variant[-1]
        score = round(feature_db.loc[(feature_db['ori_aa'] == wt_fa) &( feature_db['mut_aa'] == mut_fa)]['ln'].values[0],4)
        db.loc[db['variant'] == variant, newcolname] = score
    return db


##########################################################################

# pLDDT and pLDDTbin

##########################################################################

def parse_mmcif(uni, path):
    '''
    Extracts residue name, position, confidence metric from AlphaFold structure files
    Returns list of list [res name, pos, confidence metric]
    '''
    structure = MMCIFParser(QUIET=True).get_structure(uni, path + 'AF-' + uni + '-F1-model_v4.cif')

    aa3to1 = Bio.Data.IUPACData.protein_letters_3to1
    aa3to1_upper = {k.upper(): v.upper() for k, v in aa3to1.items()}

    io = MMCIFIO()
    chainID = list()
    for chain in structure.get_chains():
        io.set_structure(chain)
        if chain.get_id() not in chainID:
            chainID.append(chain.get_id())
    if len(chainID) > 1:
        sys.exit('more than one chain. check')

    seq, res_num, confidence_metric = [], [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                seq.append(aa3to1_upper.get(residue.resname, 'X'))
                res_num.append(residue.id[1])
                confidence_metric.append([atom.bfactor for atom in residue if atom.name == 'CA'][0])

    res_resno_cm = [[seq[i], res_num[i], confidence_metric[i]] for i in range(len(seq))]
    return res_resno_cm


def AF_cm_index(db, res_resno_cm):
    '''
    Adding pLDDT and pLDDTbin features to the protein dataframe
    As an input, it takes the protein dataframe and the output of the parse_mmcif function (list of list with residue name. position, confidence metric)
    Returns the protein dataframe with two new columns added
    '''
    db = db.copy()

    # Add confidence metric as a new column to the protein dataframe by matching res_name and pos
    db['pLDDT'] = np.NaN
    for res in res_resno_cm:
        wt, pos, cm = res[0], res[1], res[2]
        db.loc[(db['first'] == wt) & (db['pos'] == pos), 'pLDDT'] = cm

    if db['pLDDT'].isna().values.any():
        print(f'pLDDT has NaNs at positions: {db[db.pLDDT.isna()].pos.unique()}')

    # Binarize confidence metric feature
    db['pLDDT bin'] = np.NaN
    db.loc[(db['pLDDT'] >= 70), 'pLDDT bin'] = 1
    db.loc[(db['pLDDT'] < 70), 'pLDDT bin'] = 0
    return db


def add_af_confidencemetric(uni, path, db):
    print(uni, db.uniprot.unique()[0])
    res_resno_cm = parse_mmcif(uni, path)  # Get confidence metric info
    db_new = AF_cm_index(db, res_resno_cm)  # Add it to protein db as new columns
    if not np.array_equal(db_new.uniprot.unique(), db_new.uniprot.unique()):
        sys.exit('uniprots do not match')
    return db_new


##########################################################################

# colasi

##########################################################################

def first_contact_layer_atomic(jsonfile, log_file):
    '''
    Parsing the output of Arpeggio tool, which returns a .json file with all atomic interactions between residues
    The Function parses the file in order to identify list of interactions with the following filters:
        - do not take the atomic interactions between adjacent residues
        - take only interactions that are less then or equal to 5 Angstrom
        - in case of duplicated interactions / repeated couples, filter out

    Returns:
        - dff_summary_clean: table with a detailed summary of all the contacting pairs (essentially it is dataframe format of the json file output after filtering, and keeping those we want to use for future steps)
            columns: couple_id, distance, interacting_entities, res1, pos1, atom_id1, res2, pos2, atom_id2, atom_couple, r1p1, r2p2
        - contact_df: table with number of contact counts for each residue (columns:pos1, res1, counts)
        - log_file:
        -
    '''

    with open(jsonfile, 'r') as f:
        data = json.loads(f.read())
    df_json = pd.json_normalize(data)
    df_json['contact'] = df_json['contact'].apply(lambda x: x[0])

    print('No chain filters:', df_json['bgn.auth_asym_id'].unique())

    # Filter by label_comp_id so that we only have a dataframe with residues
    aa3to1 = Bio.Data.IUPACData.protein_letters_3to1
    aa3to1_upper = {k.upper(): v.upper() for k, v in aa3to1.items()}

    # Filter for contacts with water etc., we only want to keep contacts between residues
    dff = df_json.loc[(df_json['bgn.label_comp_id'].isin(list(aa3to1_upper.keys()))) &
                      (df_json['end.label_comp_id'].isin(list(aa3to1_upper.keys())))]

    # Filter for distance
    dff = dff[dff.distance <= 5]

    # Drop duplicated lines if any
    dff = dff.drop_duplicates(subset=['bgn.auth_atom_id', 'bgn.auth_seq_id', 'bgn.label_comp_id',
                                      'end.auth_atom_id', 'end.auth_seq_id', 'end.label_comp_id',
                                      'distance', 'interacting_entities'], keep="first").reset_index(drop=True)

    # Remove cases where position < 1
    dff_filter = dff[(dff['bgn.auth_seq_id'] > 0) & (dff['end.auth_seq_id'] > 0)].reset_index(drop=True)
    if len(dff_filter) - len(dff) != 0:
        print('-------- Position < 1 found... ----------\n')

    # The begin - end pair does not appear again in the reverse order
    # therefore,  parse each pair from both sides (1-2 and 2-1)
    dff_summary = pd.DataFrame()

    for i in dff_filter.index.values:
        row = dff_filter.loc[dff_filter.index == i].copy()

        # Skip contact between adjacent residues
        if np.abs(row['bgn.auth_seq_id'].values[0] - row['end.auth_seq_id'].values[0]) > 1:
            # Atom 1 and atom 2
            pair12 = row.rename(columns={'bgn.label_comp_id': 'res1',
                                         'bgn.auth_seq_id': 'pos1',
                                         'bgn.auth_atom_id': 'atom_id1',
                                         'end.label_comp_id': 'res2',
                                         'end.auth_seq_id': 'pos2',
                                         'end.auth_atom_id': 'atom_id2'})

            pair12['couple_id'] = 'index_' + str(i)
            pair12['atom_couple'] = [[pair12['atom_id1'].values[0], pair12['atom_id2'].values[0]]]

            # Atom 2 and atom 1
            pair21 = row.rename(columns={'end.label_comp_id': 'res1',
                                         'end.auth_seq_id': 'pos1',
                                         'end.auth_atom_id': 'atom_id1',
                                         'bgn.label_comp_id': 'res2',
                                         'bgn.auth_seq_id': 'pos2',
                                         'bgn.auth_atom_id': 'atom_id2'})

            pair21['couple_id'] = 'index_' + str(i)
            pair21['atom_couple'] = [[pair21['atom_id2'].values[0], pair21['atom_id1'].values[0]]]

            dff_summary = pd.concat([dff_summary, pair12, pair21])

        else:
            log_file.write(f'adjacent neighbours at index {i} : {row["bgn.auth_seq_id"].values[0]} and {row["end.auth_seq_id"].values[0]}\n')

    dff_summary = dff_summary[['couple_id', 'distance', 'contact', 'interacting_entities',
                               'res1', 'pos1', 'atom_id1', 'res2', 'pos2', 'atom_id2', 'atom_couple']]

    # Drop duplicated atomic contacts if any (could be due to alternate residues, therefore same res-pos couples with different distance are the duplicates to be removed)
    toberemoved = list(dff_summary[dff_summary.duplicated(subset=['res1', 'pos1', 'res2', 'pos2', 'atom_id1', 'atom_id2'])].couple_id.unique())

    #if len(toberemoved) > 0:
        #print('Duplicated lines found in parsed file and removed. Check.')

    dff_summary_clean = dff_summary[~dff_summary.couple_id.isin(toberemoved)]
    dff_summary_clean = dff_summary_clean.sort_values(by='pos1')

    print('Interacting entities:', dff_summary.interacting_entities.unique())
    print('Contacts:', dff_summary.contact.unique(), '\n')

    log_file.write(f'Interacting entities: {dff_summary.interacting_entities.unique()}\n')
    log_file.write(f'Contacts: {dff_summary.contact.unique()}\n')

    # For each unique pos1, count how many res1 exists. Store this information in a new df
    contact_df = dff_summary_clean.groupby(['pos1', 'res1']).size().reset_index(name='counts')

    return dff_summary_clean, contact_df, log_file


def first_contact_layer_countTables(filelist, path_files):
    '''
    Storing the outputs of the first_contact_layer_atomic function for the protein of interest
    the outputs stored in the specified path are:
    - #uniprotID_atomic.csv
    - #uniprotID_contact_pairs_atomic.csv
    '''
    for file in filelist:
        print(f'starting... {file}\n')
        jsonfile = os.path.join(path_files, file, f'AF-{file}-F1-model_v4.json')
        log_file_path = os.path.join(path_files, file, f'{file}_log.txt')

        with open(log_file_path, "w") as log_file:
            dff_summary_clean, contact_df, log_file = first_contact_layer_atomic(jsonfile, log_file)
            contact_df.to_csv(os.path.join(path_files, file, f'{file}_atomic.csv'), index=False)
            dff_summary_clean.to_csv(os.path.join(path_files, file, f'{file}_contact_pairs_atomic.csv'), index=False)
    print('Done!')



def normalize(a, b):
    return round(a / b, 3)

def add_normalized_first_contact_layer(path_files, dirlist, max_counts_df):
    '''
    Parsing the _atomic.csv files of proteins, and adding 2 new columns.
    One column is the maximum number of contacts a residue can make (coming from max_counts)
    Second column is the normalized counts value, equals to: count / max counts
    '''
    print('-- Adding normalized counts column to "_atomic.csv" file --')

    aa3to1 = Bio.Data.IUPACData.protein_letters_3to1
    aa3to1_upper = {k.upper(): v.upper() for k, v in aa3to1.items()}

    for file in dirlist:
        print(file)
        df_path = os.path.join(path_files, file, file + '_atomic.csv')
        df = pd.read_csv(df_path)

        # Convert max counts dataframe to dictionary
        max_dict = dict(zip(max_counts_df['residue'], max_counts_df['max_vals']))

        # Add maximum values from dictionary
        df['max_counts'] = df['res1'].map(max_dict)

        # Normalize
        df['colasi'] = df.apply(lambda row: normalize(row['counts'], row['max_counts']), axis=1)
        if len(df[df.colasi.isna()]) != 0:
            sys.exit('Normalized counts column has NaNs.')

        # Columns we will need to merge with FA dataset
        df['uniprot'] = file
        df['wt_pos'] = df.apply(lambda row: aa3to1_upper[row['res1']] + str(row['pos1']), axis=1)

        df.to_csv(os.path.join(path_files, file, file + '_atomic_norm.csv'), index=0)
    print('done.')



def add_colasi(db, path_files):
    '''
    Adding colasi feature to the protein db input
    Returns the protein db with a new column added
    '''
    db['wt_pos'] = db['variant'].str[:-1]
    uni = db.uniprot.unique()[0]
    db_to_merge = pd.read_csv(os.path.join(path_files, uni, uni + '_atomic_norm.csv'))
    db_protein_merged = pd.merge(db, db_to_merge[['uniprot','wt_pos', 'colasi']], on= ['uniprot','wt_pos'], how='left')
    # fill NaNs of colasi column with zero, (in some cases, no info available in the Arpeggio output for some positions)
    db_protein_merged['colasi'] = db_protein_merged['colasi'].fillna(0)
    return db_protein_merged


##########################################################################

def which_feature_pos_na(tmp, feature):
    '''
    Check if there are any NaNs in a column
    Inputs:
        - tmp: dataframe of interest
        - feature: string with the name of column in the tmp dataframe
    '''
    uni = tmp.uniprot.unique()[0]
    tmp1 = tmp[tmp[feature].isna()]
    if len(tmp1) > 0:
        print(feature)
        positions = tmp1.pos.unique()
        print(f'uniprot: {uni}, positions: {positions}, cm_index: {tmp1["pLDDT bin"].unique()}')
    else:
        print(f'no NaNs in {feature} column')


##########################################################################

# Fraction conserved 3D Neighbours

##########################################################################

def add_fraction_cons_3D_neigh(path_fa, db_protein, t1, t2):
    '''
    Capture how conserved the surrounding environment of a residue is (whether it is lining in a conserved or non-conserved region)

    - For a given wt residue, first collect the atomic contacts this residue makes
    - Collect the entropy values of these atomic contacts (Ni)
    - Using the entropy thresholds t1 and t2, count ultra-conserved (nu), well-conserved (nw), low-conserved (nl) contacts
    - Calculate frequencies fu = nu/Ni, fw = nw/Ni, fl = nl/Ni

    Returns:
        - protein dataframe with ultra conserved frequency (fu) added as a new column

    Additionally, the function stores the following dataframes in the specified folder:
        - #uniprotID__atomic_ent_freqs.csv stores the information of contact entropies, nu/nw/nl and fu/fw/fl values
        - #uniprotID_contact_pairs_atomic.csv updated the existing file with r1p1 and r2p2 columns
    '''
    # Define aa3to1_upper
    aa3to1 = Bio.Data.IUPACData.protein_letters_3to1
    aa3to1_upper = {k.upper(): v.upper() for k, v in aa3to1.items()}


    uni = db_protein.uniprot.unique()[0]

    print(uni)
    # Load contact pairs
    df_contacts = pd.read_csv(os.path.join(path_fa, uni, f'{uni}_contact_pairs_atomic.csv')).sort_values(by='pos1')
    df_contacts['r1p1'] = df_contacts.apply(lambda row: aa3to1_upper.get(row['res1'], 'X') + str(row['pos1']), axis=1)
    df_contacts['r2p2'] = df_contacts.apply(lambda row: aa3to1_upper.get(row['res2'], 'X') + str(row['pos2']), axis=1)

    # Load atomic counts
    df_counts = pd.read_csv(os.path.join(path_fa, uni, f'{uni}_atomic.csv'))
    df_counts['r1p1'] = df_counts.apply(lambda row: aa3to1_upper.get(row['res1'], 'X') + str(row['pos1']), axis=1)
    df_counts['contacts_entropies'] = np.NaN
    df_counts['nu_nw_nl'] = np.NaN
    df_counts['fu_fw_fl'] = np.NaN

    # Load entropy data
    df_entropy = pd.read_csv(os.path.join(path_fa, uni, f'{uni}_entropy.csv'))
    df_entropy_unique = df_entropy.drop_duplicates(subset=['wt_pos'], keep="first").reset_index(drop=True)

    # Copy the protein database
    db_protein = db_protein.copy()

    for wtpos in df_contacts['r1p1'].unique():
        tocheck, entropy_list = [], []  # Collect tuples of r2p2 & collect entropy values
        neighbors = df_contacts[df_contacts['r1p1'] == wtpos]['r2p2'].values

        for n in neighbors:
            entropy_row = df_entropy_unique[df_entropy_unique['wt_pos'] == n]
            if not entropy_row.empty:
                entropy = entropy_row["Shannon's entropy"].values[0]
                entropy_list.append(entropy)
                tocheck.append((n, entropy))
            else:
                entropy_list.append(0)
                tocheck.append((n, 0))
                print('Missing entropy data for:', n)

        # Ni = number of atomic contacts of residue i  where Ni = nu+nw+nl
        Ni = len(entropy_list)  # This number should coincide with the value in df_counts

        if df_counts[df_counts['r1p1'] == wtpos]['counts'].values[0] != Ni:
            sys.exit('Number of contacts do not match with df_counts dataframe.')

        #nu : number of contacts coming from ultra-conserved residues  	 entropy < 4.12/3
        #nw : number of contacts coming  from well-conserved residues 	 4.12/3 > entropy > 2*4.12/3
        #nl :  number of contacts coming  from low-conserved residues  	 entropy > 4.12/3

        nu, nw, nl = 0, 0, 0  # Number of ultra-conserved, well-conserved, low-conserved
        for val in entropy_list:
            if val <= t2:
                nu += 1
            elif t2 < val <= t1:
                nw += 1
            elif val > t1:
                nl += 1
        if (nu + nw + nl) != Ni:
            sys.exit('nu + nw + nl != Ni')

        fu, fw, fl = nu / Ni, nw / Ni, nl / Ni


        db_protein.loc[db_protein['wt_pos'] == wtpos, 'fraction cons. 3D neighbor'] = round(fu, 3)

        #db_protein.loc[db_protein['wt_pos'] == wtpos, 'ultra_cons_ent'] = round(fu, 3)
        #db_protein.loc[db_protein['wt_pos'] == wtpos, 'well_cons_ent'] = round(fw, 3)
        #db_protein.loc[db_protein['wt_pos'] == wtpos, 'low_cons_ent'] = round(fl, 3)

        df_counts.loc[df_counts['r1p1'] == wtpos, 'contacts_entropies'] = str(tocheck)
        df_counts.loc[df_counts['r1p1'] == wtpos, 'nu_nw_nl'] = str([nu, nw, nl])
        df_counts.loc[df_counts['r1p1'] == wtpos, 'fu_fw_fl'] = str([round(fu, 3), round(fw, 3), round(fl, 3)])

    # Save updated dataframes
    df_counts.to_csv(os.path.join(path_fa, uni, f'{uni}__atomic_ent_freqs.csv'), index=False)
    df_contacts.to_csv(os.path.join(path_fa, uni, f'{uni}_contact_pairs_atomic.csv'), index=False)

    return db_protein


##########################################################################

# Fanc & Fbnc

##########################################################################

def add_exInd(db):
    '''
    Adds the information whether a residue is located in an exposed or buried region based on the colasi feature (which represents the normalized number of atomic contacts the residue makes)
    Returns the protein dataframe with "exInd" column added
    '''
    db['exInd'] = np.NaN
    db.loc[(db['colasi'] <= 0.5), 'exInd'] = 1 # exposed
    db.loc[(db['colasi']  > 0.5), 'exInd'] = 0 # buried
    if len(db[db.exInd.isna()]) > 0:
        sys.exit('exInd column has NaN.')
    return db

def add_exInd_feature(path_fa, db_protein):
    uni = db_protein.uniprot.unique()[0]
    print(uni)

    # add to main dataframe
    db_merged = add_exInd(db_protein)

    # add to atomic_norms files
    db_to_merge = pd.read_csv(os.path.join(path_fa, uni, f'{uni}_atomic_norm.csv'))
    db_to_merge_merged = add_exInd(db_to_merge)
    db_to_merge_merged.to_csv(os.path.join(path_fa, uni, f'{uni}_atomic_norm.csv'), index=0)

    print('exposed/buried binary info added to \n-  "protein dataframe" & "_atomic_norm.csv" files.')
    return db_merged


def add_fanc_fbnc(path_fa, db_protein):
    '''
    Calculating the fraction of the first contact layer of residue i with median entropy

    e.g. for an exposed residue:
    - Find the exposed residues (using exInd column)
    - For a given residue i, get the list of contacting residues (ncl)
    - From this list of contacts, select those atomic contacts that are exposed (lexcl)
    - From the lexcl list, select those that are MORE CONSERVED than the median exposed entropy of that protein
    - Compute the final feature fanc = lexclco / ncl
    Repeat the same process for the buried residues

    Returns the db_protein dataframe with two new columns fanc and fbnc added
    # Note: the numerical values of ncl, lexcl etc. are collected in the "cheatsheet_dict" dictionary, which can be saved as a dataframe for a more detailed look (currently commented so it is not stored)
    '''
    aa3to1 = Bio.Data.IUPACData.protein_letters_3to1
    aa3to1_upper = {k.upper(): v.upper() for k, v in aa3to1.items()}

    uni = db_protein.uniprot.unique()[0]
    print(uni)
    # Load normalized atomic counts
    df_counts_norm = pd.read_csv(os.path.join(path_fa, uni, f'{uni}_atomic_norm.csv'))

    # Load entropy data
    df_entropy = pd.read_csv(os.path.join(path_fa, uni, f'{uni}_entropy.csv'))
    df_entropy_unique = df_entropy.drop_duplicates(subset=['wt_pos'], keep="first").reset_index(drop=True)

    # Load contact pairs
    df_contacts = pd.read_csv(os.path.join(path_fa, uni, f'{uni}_contact_pairs_atomic.csv')).sort_values(by='pos1')

    # Get exposed median entropy
    df_exposed = df_counts_norm[df_counts_norm.exInd == 1]
    if len(df_exposed) == 0:
        sys.exit('No exposed residues found in this protein.')

    df_exposed_entropy = dict(df_entropy_unique[df_entropy_unique.wt_pos.isin(df_exposed.wt_pos.unique())][['wt_pos', "Shannon's entropy"]].values)
    median_entropy_exposed = round(np.median(list(df_exposed_entropy.values())), 3)

    # Get buried median entropy
    df_buried = df_counts_norm[df_counts_norm.exInd == 0]
    if len(df_buried) == 0:
        sys.exit('No buried residues found in this protein.')

    df_buried_entropy = dict(df_entropy_unique[df_entropy_unique.wt_pos.isin(df_buried.wt_pos.unique())][['wt_pos', "Shannon's entropy"]].values)
    median_entropy_buried = round(np.median(list(df_buried_entropy.values())), 3)

    print('.....median entropy exposed:', median_entropy_exposed)
    print('.....median entropy buried:', median_entropy_buried)

    db_protein['fanc'] = np.NaN
    db_protein['fbnc'] = np.NaN

    cheatsheet_dict = {}

    for wtpos in df_counts_norm['wt_pos'].unique():
        minidict = {}

        # Number of contacts residue wtpos makes
        tmp = df_contacts[df_contacts['r1p1'] == wtpos]
        ncl = len(tmp)

        minidict['neighbors'] = list(tmp.r2p2.values)
        minidict['ncl'] = ncl
        if ncl == 0:
            sys.exit('ncl is zero.')

        if df_counts_norm[df_counts_norm.wt_pos == wtpos]['counts'].values[0] != ncl:
            sys.exit(f'Number of contacts for residue {wtpos} is not correct. Check df_contacts and df_counts_norm.')

        # lexcl: number of exposed/buried residues in ncl
        lexcl_e, lexcl_b = [], []

        for n in tmp['r2p2'].values:
            if n in df_exposed['wt_pos'].values:
                lexcl_e.append(n)
            elif n in df_buried['wt_pos'].values:
                lexcl_b.append(n)

        if len(lexcl_e) + len(lexcl_b) != ncl:
            sys.exit('lexcl_e + lexcl_b != ncl')

        # lexclco: from lexcl, select those residues that are more conserved than the median exposed/buried
        lexclco_e = [i for i in lexcl_e if df_exposed_entropy[i] < median_entropy_exposed]
        lexclco_b = [i for i in lexcl_b if df_buried_entropy[i] < median_entropy_buried]

        # fexclco: lexclco / ncl
        fexclco_e = round(len(lexclco_e) / ncl, 3)
        fexclco_b = round(len(lexclco_b) / ncl, 3)

        db_protein.loc[db_protein.wt_pos == wtpos, 'fanc'] = fexclco_e
        db_protein.loc[db_protein.wt_pos == wtpos, 'fbnc'] = fexclco_b

        minidict.update({
            'lexcl_e': lexcl_e,
            'excl_e': len(lexcl_e),
            'lexclco_e': lexclco_e,
            'fexclco_e': fexclco_e, #fanc
            'lexcl_b': lexcl_b,
            'excl_b': len(lexcl_b),
            'lexclco_b': lexclco_b,
            'fexclco_b': fexclco_b #fbnc
        })

        cheatsheet_dict[wtpos] = minidict

    # save this file to check manually the counts and calculations
    #cheatsheet = pd.DataFrame(cheatsheet_dict).T
    #cheatsheet.to_csv(os.path.join(path_fa, uni, f'{uni}_cheatsheet_fanc_fbnc.csv'), index=False)

    return db_protein


##########################################################################

# MJ Potential

##########################################################################


def cm_index_parse(uni, path):
    '''
    extracts residue name, position, confidence metrix
    returns list of list [res name, pos, confidence metric]
    '''
    structure = MMCIFParser(QUIET=1).get_structure(uni, path+'/'+uni+'/'+'AF-' + uni + '-F1-model_v4.cif')

    aa3to1 = Bio.Data.IUPACData.protein_letters_3to1
    aa3to1_upper = {k.upper():v.upper() for k,v in aa3to1.items()}

    io = MMCIFIO()
    chainID = list()
    for chain in structure.get_chains():
        io.set_structure(chain)
        if chain.get_id() not in chainID:
            chainID.append(chain.get_id())
    if len(chainID) > 1:
        sys.exit('more than one chain. check')

    for model in structure:
        for chain in model:
            seq, res_num, confidence_metric = [],[],[]
            for residue in chain:
                seq.append(aa3to1_upper[residue.resname])
                res_num.append(residue.id[1])
                confidence_metric.append([atom.bfactor for atom in residue if atom.name=='CA'][0])

    res_resno_cm = [[seq[i], res_num[i], confidence_metric[i]] for i in range(len(seq))]
    return res_resno_cm



def create_pdb_table(file,uni, path_af):
    '''
    Taking the x,y,z coordinates of the side chain atoms for a given protein structure
    If it is a glycine, take C-alpha coordinates
    Returns a table with residue, position, x,y,z coordinates, pLDDT and pLDDTbin columns
    This table is stored as "side_chain_table.csv" with the "mj_potential_prep" function
    '''
    backbone_atoms = ['CA','C','O','N']
    list_of_pdb = []
    parser = MMCIFParser()
    s = parser.get_structure(file, path_af + uni + '/'+file+'.cif')
    io = MMCIFIO()
    for model in s.get_list():
        for chain in model.get_list():
            for residue in chain.get_list():
                resname = residue.get_resname()
                atoms = residue.get_atoms()
                #print(resname, residue.get_id()[1])

                if resname != 'GLY':
                    list_of_atoms = []
                    for atom in atoms:
                        if atom.get_name() not in backbone_atoms:
                            #print(atom.get_name(), atom.get_coord())
                            list_of_atoms.append([atom.get_name(), list(atom.get_coord())])

                    list_of_pdb.append([resname, residue.get_id()[1], list_of_atoms])

                elif resname == 'GLY':
                    list_of_atoms = []
                    for atom in atoms:
                        if atom.get_name() in ['CA']:
                            #print(atom.get_name(), atom.get_coord())
                            list_of_atoms.append([atom.get_name(), list(atom.get_coord())])

                    list_of_pdb.append([resname, residue.get_id()[1], list_of_atoms])

                #print('__________________________')

    pdb_table = pd.DataFrame()
    for each in list_of_pdb:
        #print(each[0], each[1])
        atomic_list = each[2]
        tmp = pd.DataFrame(atomic_list, columns=['atom','coords'])
        tmp[['x_coord','y_coord','z_coord']] = pd.DataFrame(tmp.coords.tolist(), index= tmp.index)
        tmp[['x_coord','y_coord','z_coord']] = tmp[['x_coord','y_coord','z_coord']].apply(lambda x: round(x,4), axis=1)
        tmp['residue'] = each[0]
        tmp['pos'] = each[1]
        pdb_table = pd.concat([pdb_table,tmp])

    res_resno_cm=cm_index_parse(uni, path_af)

    pdb_table = pdb_table.copy()
    pdb_table['wt'] = pdb_table['residue'].apply(lambda x: seq1(x))

    for i in range(len(res_resno_cm)):
        wt, pos, cm = res_resno_cm[i][0], res_resno_cm[i][1],res_resno_cm[i][2]
        pdb_table.loc[(pdb_table['wt'] == wt) & (pdb_table['pos'] == pos), 'pLDDT'] = cm

    pdb_table = pdb_table.copy()
    pdb_table.loc[(pdb_table['pLDDT'] >= 70), 'pLDDT bin'] = 1
    pdb_table.loc[(pdb_table['pLDDT'] < 70), 'pLDDT bin'] = 0

    pdb_table = pdb_table[['residue','pos','atom','coords','x_coord','y_coord', 'z_coord','pLDDT', 'pLDDT bin']]

    return pdb_table



def average_side_chains(pdb_table):
    '''
    For a given side chains table, take the average of each column to define side-chain centers
    Returns a table with the residue, position, x_mean, y_mean, z_mean columns

    This table is stored as "avg_side_chain_table.csv" with the "mj_potential_prep" function
    '''
    avg_pdb_table = pd.DataFrame()
    for position in pdb_table.pos.unique():
        tmp = pdb_table[pdb_table.pos==position]
        tmp_avg = pd.DataFrame(tmp[['x_coord', 'y_coord','z_coord']].mean()).T.rename(columns={'x_coord':'x_mean',
                                                                                               'y_coord':'y_mean',
                                                                                               'z_coord':'z_mean'})
        tmp_avg['residue'] = tmp.residue.unique()[0]
        tmp_avg['pos'] = position
        tmp_avg['pLDDT bin'] = tmp['pLDDT bin'].unique()[0]
        tmp_avg['pLDDT'] = tmp['pLDDT'].unique()[0]
        avg_pdb_table = pd.concat([avg_pdb_table, tmp_avg])

    avg_pdb_table = avg_pdb_table[['residue','pos','x_mean','y_mean','z_mean', 'pLDDT', 'pLDDT bin']].reset_index(drop=True)
    avg_pdb_table[['x_mean','y_mean','z_mean']] = avg_pdb_table[['x_mean','y_mean','z_mean']].apply(lambda x: round(x,4), axis=1).copy()

    return avg_pdb_table


def contact_from_average(avg_pdb_table):
    '''
    Using the table with average x,y,z positions of the side chain atoms of each residue (for a given protein),
    create contacting residues table by calculating the euclidean distance between:
        - the wt residue x_mean,y_mean,z_mean, and
        - the contact residue x_mean,y_mean,z_mean values
    Contact definition: distance between residue side chain centers

    For the final selection of "contacts", apply the following filters:
        - filter1: do not take sequence adjacent neighbours as contact
        - filter2: take only those that have distance smaller than or equal to 6.5
        - filter3: count only if it is a high quality residue (pLDDT bin = 1)

    Returns a table with residue, position, contact residue, contact position, distance, pLDDT and pLDDT bin columns

    This table is stored as "contact_avg_side_chains.csv" with the "mj_potential_prep" function

    '''
    contact_table = pd.DataFrame()
    collect = []

    for position in avg_pdb_table.pos.unique():

        # find contacts of this residue
        main = avg_pdb_table[avg_pdb_table.pos==position]
        main_residue = main.residue.unique()[0]
        main_dist = list(main[['x_mean', 'y_mean','z_mean']].values[0])

        adjacent_positions = [position-1, position+1]
        # iterate rest of the residues
        rest = avg_pdb_table[avg_pdb_table.pos!=position]

        # FILTER 1: remove adjacent positions from this list
        rest = rest[~rest.pos.isin(adjacent_positions)].reset_index(drop=True).copy()

        for index, row in rest.iterrows():

            # FILTER 2: Take only high quality residues
            if row['pLDDT bin'] == 1:
                dist_rest = [row['x_mean'], row['y_mean'], row['z_mean']]
                dist = round(distance.euclidean(main_dist, dist_rest),3)

                # FILTER 3: if this distance is smaller than 6.5 we add it to contact_table
                if dist <= 6.5:
                    collect.append([main_residue,position,row['residue'],row['pos'],dist, row['pLDDT'], row['pLDDT bin']])
            elif row['pLDDT bin'] == 0:
                pass

    contact_table = pd.DataFrame(collect)
    contact_table.columns=['residue','pos','contact_residue','contact_pos','distance', 'pLDDT', 'pLDDT bin']

    return contact_table


def mj_potential_prep(uni, path_af):

    file = f'AF-{uni}-F1-model_v4'
    print('uniprot: ', uni, '//  structure file name: ', file)
    pdb_table = create_pdb_table(file, uni, path_af)
    avg_pdb_table = average_side_chains(pdb_table)
    contact_table = contact_from_average(avg_pdb_table)

    pdb_table.to_csv(os.path.join(path_af, uni, f'side_chain_table.csv'), index=False)
    avg_pdb_table.to_csv(os.path.join(path_af, uni, f'avg_side_chain_table.csv'), index=False)

    contact_table['wt'] = contact_table['residue'].apply(lambda x: seq1(x))
    contact_table['wt_pos'] = contact_table['wt'] + contact_table['pos'].astype(str)
    contact_table['contact_wt'] = contact_table['contact_residue'].apply(lambda x: seq1(x))
    contact_table = contact_table[['residue', 'wt', 'pos', 'wt_pos', 'contact_residue','contact_wt','contact_pos', 'distance','pLDDT bin']]
    contact_table.to_csv(os.path.join(path_af, uni, f'contact_avg_side_chains.csv'), index=False)

    print(f'outputs: created in {path_af}uni/:\n- side_chain_table.csv\navg_side_chain_table.csv\n-contact_avg_side_chains.csv')
    return None





def get_score(mj_table, val1, val2):
    '''
    Parses the mj_table to get the value for the given wt-mt pair
    '''
    mj_table = mj_table.fillna('no')
    score = [a for a in [mj_table[mj_table.index==val1][val2].values[0], mj_table[mj_table.index==val2][val1].values[0]] if a != 'no'][0]
    return score



def add_MJ_potential(db,contact_table,mj_table):
    '''
    Applying the formula: Dei,j = sum_j(emut,j - enat,i), where:
        - i = native residue
        - j = contact residue
        - mut = variant version of i
        enat,i = sum of all contacts around native
        emut,j = sum of all contacts around mutant

    Returns protein dataframe with "M.J potential" column added
    '''
    db_merged = pd.DataFrame()

    for wtpos in db['wt_pos'].unique():

        db_wtpos_tmp = db[db.wt_pos ==wtpos].copy()
        native = db_wtpos_tmp['first'].values[0]

        # what contacts is wtpos making
        contacts = list(contact_table[contact_table.wt_pos==wtpos]['contact_wt'].values)

        if len(contacts) > 0:

            # what is the score for the native
            scores = []
            for c in contacts:
                score = get_score(mj_table,native,c)
                scores.append(score)

            e_natj = sum(scores)
            db_wtpos_tmp['e_native'] = e_natj


            # what is the score for the mutant
            scores_variant = dict()
            for i,row in db_wtpos_tmp.iterrows():

                mutant = row['second']
                scores = []

                for c in contacts:
                    score = get_score(mj_table,mutant,c)
                    scores.append(score)

                e_mutj = sum(scores)

                scores_variant[mutant] = e_mutj

            for k,v in scores_variant.items():
                db_wtpos_tmp.loc[db_wtpos_tmp.second == k, 'e_mutant'] = v
                db_wtpos_tmp.loc[db_wtpos_tmp.second == k, 'mj_potential'] = v-e_natj

        else:
            db_wtpos_tmp['mj_potential'] = 0
            db_wtpos_tmp['e_native'] = 0
            db_wtpos_tmp['e_mutant'] = 0

        db_merged = pd.concat([db_merged, db_wtpos_tmp])

    db_merged.loc[(db_merged['pLDDT bin']==0) & (db_merged['mj_potential'].isna()), 'mj_potential'] = 0

    db_merged.rename(columns={'mj_potential':'M.J. potential'}, inplace=True)
    del db_merged['e_native']
    del db_merged['e_mutant']
    return db_merged


##########################################################################

# Accessibility dependent volume term

##########################################################################

def add_acc_dependent_vol(path_af, db_protein, max_counts):
    '''
    Applying the formula: [mxng(mut)-mxng(nat)].[ng(nat)/mxng(nat)], where
        - mxng(mut)= maximum possible contact for the mutant
        - mxng(nat) = maximum possible contact for the native
        - ng(nat) = number of contacts the native has
    Returns protein dataframe with 'access.dependent vol.' column added
    '''

    uni = db_protein.uniprot.unique()[0]
    print(uni)

    uni_atomic_norm = pd.read_csv(os.path.join(path_af, uni, f'{uni}_atomic_norm.csv'), index_col=0)

    # Add the excluded volume feature
    for i, row in db_protein.iterrows():
        max_cont_nat = max_counts[max_counts['res_one_letter'] == row['first']]['max_vals'].values
        max_cont_mut = max_counts[max_counts['res_one_letter'] == row['second']]['max_vals'].values

        if len(max_cont_nat) == 0 or len(max_cont_mut) == 0:
            print(f'Max contact values not found for residue {row["first"]} or {row["second"]}')
            exc_vol = np.NaN
        else:
            max_cont_nat = max_cont_nat[0]
            max_cont_mut = max_cont_mut[0]

            if row['wt_pos'] in uni_atomic_norm['wt_pos'].values:
                cont_nat = uni_atomic_norm[uni_atomic_norm['wt_pos'] == row['wt_pos']]['counts'].values[0]
                exc_vol = (max_cont_mut - max_cont_nat) * (cont_nat / max_cont_nat)
            else:
                exc_vol = np.NaN

        db_protein.loc[i, 'access.dependent vol.'] = round(exc_vol, 3) if not pd.isna(exc_vol) else 0

    return db_protein


##########################################################################

# Laar

##########################################################################

def add_laar(path_file, db_protein):
    # please refer to our QAFI paper to follow how feature_laar.csv data was calculated
    # Credit for calculation of this feaure: Natàlia Padilla Sirera https://github.com/NataliaSirera
    print(db_protein.uniprot.unique()[0])
    laar_df = pd.read_csv('data/feature_laar.csv')
    db_protein['e_b_status'] = np.NaN
    db_protein.loc[db_protein.colasi <= 0.5, 'e_b_status'] = 'exposed'
    db_protein.loc[db_protein.colasi > 0.5, 'e_b_status'] = 'buried'
    db_merged = db_protein.merge(laar_df[['first', 'second', 'e_b_status', 'laar']],
                                        on = ['first', 'second', 'e_b_status'], how='left')

    del db_merged['e_b_status']
    return db_merged
