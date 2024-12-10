# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 20:28:19 2024

@author: apurv
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.rdFingerprintGenerator
import sklearn.metrics as metrics
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.gaussian_process.kernels import RBF, Matern,WhiteKernel, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn import preprocessing as p
from rdkit.Chem import Descriptors
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import random
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import csv
from rdkit.Chem import Draw
from matplotlib.lines import Line2D

def draw(smile, fn):
    m = Chem.MolFromSmiles(smile)
    img = Draw.MolToImage(m)
    fname = '../datasets/OROP313/rop313/orop/'+str(fn)+'/reduced_img.png'
    img.save(fname)
    
def smilesToMorgan(row, col='smiles', radius=2, nBits=512,bit_info=False):
    """
        Convert smiles to Morgan fingerprint.
        :param smiles: str
        :return: fingerprint as bit-vector
    """
    smiles = row[col]
    mol = MolFromSmiles(smiles)
    #print(smiles)
    
    info={}
    fp = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits,bitInfo=info)
    if bit_info:
        return pd.Series([fp,info])

    return fp

def addMorgan(df, col='smiles', radius=2, nBits=512):
    return df.apply(smilesToMorgan, args=(col, radius, nBits,False), axis=1)

def addMorganInfo(df, col='smiles', radius=2, nBits=512):
    return df.apply(smilesToMorgan, args=(col, radius, nBits,True), axis=1)

def concatMorganFps(row, col1, col2):
    fp1 = row[col1]
    fp2 = row[col2]
    n1 = fp1.GetNumBits()
    n2 = fp2.GetNumBits()
    # dummy fingerprint
    mol = MolFromSmiles("c1ccccc1O")
    fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n1 + n2)
    for indx, v in enumerate(fp1):
        fp[indx] = v
    for indx, v in enumerate(fp2):
        fp[indx + n2] = v
    return fp

def addMorganDoubleFps(df, col1='left_morgan', col2='right_morgan'):
    return df.apply(concatMorganFps, args=(col1, col2), axis=1)

def get_morgan(df, feat, rad):
    rows = df.shape[0]
    df_morgan = pd.DataFrame(columns=['morgan_l'], index=range(rows))
    df_morgan['morgan_l'] = addMorgan(df,col='LHS Compound',nBits=feat,radius=rad)
    df_morgan['morgan_r'] = addMorgan(df,col='RHS Compound',nBits=feat,radius=rad)
    df_morgan['morgan'] = addMorganDoubleFps(df_morgan, col1='morgan_l', col2='morgan_r')
    X_morgan = np.array([np.array(i) for i in df_morgan['morgan'].values])
    return X_morgan

def get_mol_desc(row, col, debug=False):
    m = row[col]
    desc = getMolDescriptors(m)
    return desc

def get_mol(row, col, debug=False):
    smile = row[col]
    m = Chem.MolFromSmiles(smile)
    return m

def getMolDescriptors(mol, missingVal=None):
    ''' calculate the full list of descriptors for a molecule
    
        missingVal is used if the descriptor cannot be calculated
    '''
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res

def get_desc_array(df):
    lhs_mols = df.apply(get_mol, args=("LHS Compound", False), axis=1)
    df['lhs_mols'] = lhs_mols
    rhs_mols = df.apply(get_mol, args = ('RHS Compound', False), axis=1)
    df['rhs_mols'] = rhs_mols
    lhs_desc = df.apply(get_mol_desc, args = ('lhs_mols', False), axis=1)
    rhs_desc = df.apply(get_mol_desc, args = ('rhs_mols', False), axis=1)
    lhs_desc = lhs_desc.values.tolist()
    rhs_desc = rhs_desc.values.tolist()
    lhs_X = pd.DataFrame(lhs_desc).values
    rhs_X = pd.DataFrame(rhs_desc).values
    X = np.concatenate((lhs_X, rhs_X), axis=1)
    X = np.apply_along_axis(myscaler, 0, X)
    X = X.squeeze()
    return X

def myscaler(a):
    a = a.reshape(-1,1)
    min_max_scaler = p.MinMaxScaler()
    nn = min_max_scaler.fit_transform(a)
    return nn

def update_qp(orig_qp_f, redox_expr_f):
    df_expr_redox = pd.read_csv(redox_expr_f)
    df_qp = pd.read_csv(orig_qp_f)
    comp_l = []
    for index, row in df_qp.iterrows():
        mol_id = row['molecule']
        if '_' in mol_id:
            smiles = df_expr_redox[df_expr_redox['RHS Id'] == mol_id]
            smiles = smiles['RHS Compound'].values[0]
        else:
            smiles = df_expr_redox[df_expr_redox['LHS Id'] == int(mol_id)]
            smiles = smiles['LHS Compound'].values[0]
        comp_l.append(smiles)
    
    df_qp.insert(1, "Compound", comp_l, True)
        
    df_qp.to_csv(orig_qp_f)
            
def get_qp(df_qp_in, df, feat):
    new_size = feat * 2
    retarr = np.empty((1,new_size), int)
    for index, row in df.iterrows():
        lhs_id = row['LHS Compound']
        sel_lhs_df = df_qp_in[df_qp_in['Compound'] == lhs_id]     
        lhs_vals = np.array(sel_lhs_df.iloc[0,2:])
        # lhs_vals = np.array(sel_lhs_df.iloc[:,2:])
        # lhs_vals = np.mean(lhs_vals, axis=0)
        lhs_vals = lhs_vals.reshape(1,-1)
        rhs_id = row['RHS Compound']
        sel_rhs_df = df_qp_in[df_qp_in['Compound'] == rhs_id]     
        rhs_vals = np.array(sel_rhs_df.iloc[0,2:])
        # rhs_vals = np.array(sel_rhs_df.iloc[:,2:])
        # rhs_vals = np.mean(rhs_vals, axis=0)
        rhs_vals = rhs_vals.reshape(1,-1)
        react_qp = np.concatenate((lhs_vals, rhs_vals), axis=1)
        retarr = np.append(retarr, react_qp, axis=0)
        pass
    
    retarr = retarr[1:,:]
    retarr1 = np.apply_along_axis(myscaler, 0, retarr)
    retarr1 = retarr1.squeeze()
    return retarr1

def get_dft(df_dft):
    np_dft = df_dft.to_numpy()
    np_dft = np_dft[:,2:]
    new_dft = np.apply_along_axis(myscaler, 0, np_dft)
    new_dft = new_dft.squeeze()
    return new_dft

def calib_gpr(Xtrain, Xtest, ytrain, ytest, Xexpr=[], yexpr=[], exprFlag=False, cvFlag=False):
    
    yexpr_pred =[]
    ypred = []
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + \
            WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)) 
    
    model = GaussianProcessRegressor(
            normalize_y=True, kernel=kernel, alpha=0.0,
            n_restarts_optimizer=2)
    model.fit(Xtrain,ytrain)
    print(model.kernel_)
    
    if not cvFlag:
        if not pred_only:
            ypred = model.predict(Xtest)
        if exprFlag and type(Xexpr) != list:
            Xtotal = np.concatenate((Xtrain, Xtest), axis=0)
            ytotal = np.concatenate((ytrain, ytest), axis=0)
            model.fit(Xtotal, ytotal)
            yexpr_pred = model.predict(Xexpr)
    else:
        model = GaussianProcessRegressor(
                normalize_y=True, kernel=model.kernel_, alpha=0.0,optimizer=None)
        
        cv = KFold(n_splits=5)
        if not pred_only:
            ypred = model_selection.cross_val_predict(model, Xtrain, ytrain, cv=cv)
            ypred = model_selection.cross_val_predict(model, Xtest, ytest, cv=cv)
        
        if exprFlag and type(Xexpr) != list: #empty list of not testing, else ndarray
            yexpr_pred = model_selection.cross_val_predict(model, Xexpr, yexpr, cv=cv)

    return ypred, yexpr_pred

def reduce_list(l, remove=[]):
    for item in remove:
        try:
            l.remove(item)
        except:
            pass

def make_scatter_plot(x, y, r2, label_x, label_y, file, color, legend_elements, annotate=False):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    plt.figure()
    legend = '$R^2=%.3f$' % r2
    plt.scatter(x, y, s=20, c=color)
    plt.plot([xmin, xmax], [xmin, xmax], ls='--',
             c='k', alpha=0.5, lw=3, label='')
    plt.xlabel(label_x, fontsize=18)
    plt.ylabel(label_y, fontsize=18)
    #plt.legend(loc='upper left', fontsize=18)
    #plt.text(xmin, ymax-int(0.005*(ymax-ymin)), legend, fontsize=18)
    plt.legend(handles=legend_elements, loc='upper left', title=legend)

    if annotate:
        for idx, x_val in enumerate(x):
           plt.text(x[idx]+10.0, y[idx]+25.0, str(idx+1), fontsize=9) 
    plt.savefig(file, dpi=300.0, bbox_inches='tight')
    
if __name__ == "__main__":
    morgan_feat = 4096
    morgan_rad = 4
    qp_feat = 51
    #dataset = 'benzo_prince'
    #dataset = 'naphtho_prince'
    #dataset = 'OROP313'
    dataset = 'OROP313'
    remove_entry_benzo = [6,7,13,16,31]
    remove_entry_naphtho = []
    remove_entry_combined_quinone = [6,7,13,16,31]
    remove_entry_OROP313 = [6,7,13,14,15,50]
    if dataset == 'benzo_prince':
        remove_entry = remove_entry_benzo
    elif dataset == 'naphtho_prince':
        remove_entry = remove_entry_naphtho
    elif dataset == 'OROP313':
        remove_entry = remove_entry_OROP313
    elif dataset == 'combined_quinone':
        remove_entry = remove_entry_combined_quinone
        
    data_dir = '../datasets/'
    dft_ready = True
    dft_ready_expr = True
    expr_test = True
    shift_expr = -500
    pred_only = False
    expr_redox_f = data_dir + dataset + '/redox_expr.csv'
    dft_energy_f = data_dir + dataset + '/dft_energy.csv'
    qp_f = data_dir + dataset + '/qikprop.csv'
    colors = ['red'] * 4 + ['goldenrod'] * 7 + ['limegreen'] * 3    
    legend_elements2 = [Line2D([0], [0], marker='o', color='w', label='Benzoquinones',
                          markerfacecolor='red', markersize=5),
                   Line2D([0], [0], marker='o', color='w', label='Naphthoquinones',
                          markerfacecolor='goldenrod', markersize=5)]
    legend_elements3 = [Line2D([0], [0], marker='o', color='w', label='Benzoquinones',
                          markerfacecolor='red', markersize=5),
                   Line2D([0], [0], marker='o', color='w', label='Naphthoquinones',
                          markerfacecolor='goldenrod', markersize=5),
                   Line2D([0], [0], marker='o', color='w', label='Anthraquinones',
                          markerfacecolor='limegreen', markersize=5)]
    
    #update_qp(qp_f, expr_redox_f)
          
    df_redox = pd.read_csv(expr_redox_f)
    df_qp = pd.read_csv(qp_f)
    if dft_ready:
        df_dft = pd.read_csv(dft_energy_f)
    
    rows = df_redox.shape[0]
    train_ratio = 0.9
    train_size = int(rows * train_ratio)
    X_morgan = get_morgan(df_redox, morgan_feat, morgan_rad)
    X_desc = get_desc_array(df_redox)
    X_qp = get_qp(df_qp, df_redox, qp_feat)
    if dft_ready:
        X_dft = get_dft(df_dft)
    else:
        X_dft = []

    if expr_test:
        dataset_test = 'test_li'
        expr_redox_f = data_dir + dataset_test + '/redox_expr.csv'
        dft_energy_f = data_dir + dataset_test + '/dft_energy.csv'
        qp_f = data_dir + dataset_test + '/qikprop.csv'
        
        df_redox_expr = pd.read_csv(expr_redox_f)
        df_qp_expr = pd.read_csv(qp_f)
        if dft_ready_expr:
            df_dft_expr = pd.read_csv(dft_energy_f)
        
        X_morgan_expr = get_morgan(df_redox_expr, morgan_feat, morgan_rad)
        X_desc_expr = get_desc_array(df_redox_expr)
        X_qp_expr = get_qp(df_qp_expr, df_redox_expr, qp_feat)
        if dft_ready_expr:
            X_dft_expr = get_dft(df_dft_expr)
        else:
            X_dft_expr = []
        y_expr=df_redox_expr['Experimental Redox'].values
    else:
        y_expr = []
    
    # if dataset == 'benzo_prince':
    #     #train = list(range(91))
    #     random.seed(1967) #2024, 2020, 2003, 2000, 1967
    #     train = random.sample(list(range(rows)), train_size)        
    # elif dataset == 'naphtho_prince':
    #     #train = list(range(train_size))
    #     random.seed(2024) #2024, 2020, 2003, 2000, 1967
    #     train = random.sample(list(range(rows)), train_size)
    # elif dataset == 'OROP313':
    #     random.seed(2024) #2024, 2020, 2003, 2000, 1967
    #     train = random.sample(list(range(rows)), train_size)
    random.seed(2000) #2024, 2020, 2003, 2000, 1967
    train = random.sample(list(range(rows)), train_size)        
    
    
    # tests_dict_benzo = {"Morgan": [X_morgan], "Desc": [X_desc], "ADME": [X_qp], "DFT": [X_dft],
    #               "Desc_ADME": [X_desc, X_qp], "DFT_ADME": [X_dft, X_qp], "Desc_DFT": [X_desc, X_dft],
    #               "Desc_ADME_DFT": [X_desc, X_qp, X_dft]}

    if expr_test:
        tests_dict_expr = {"Morgan": [X_morgan_expr], "Desc": [X_desc_expr], "ADME": [X_qp_expr], "DFT": [X_dft_expr],
                  "Desc_ADME": [X_desc_expr, X_qp_expr], "DFT_ADME": [X_dft_expr, X_qp_expr], "Desc_DFT": [X_desc_expr, X_dft_expr],
                  "Desc_ADME_DFT": [X_desc_expr, X_qp_expr, X_dft_expr]}
    else:
        tests_dict_expr = {}
    
    #tests_dict_naphtho = {"Morgan": [X_morgan], "Desc": [X_desc], "ADME": [X_qp],
                 # "Desc_ADME": [X_desc, X_qp]}
    # tests_dict_naphtho = {"Morgan": [X_morgan], "Desc": [X_desc], "ADME": [X_qp], "DFT": [X_dft],
    #               "Desc_ADME": [X_desc, X_qp], "DFT_ADME": [X_dft, X_qp], "Desc_DFT": [X_desc, X_dft],
    #               "Desc_ADME_DFT": [X_desc, X_qp, X_dft]}
    
    # tests_dict_OROP313 = {"Morgan": [X_morgan], "Desc": [X_desc], "ADME": [X_qp], "DFT": [X_dft],
    #               "Desc_ADME": [X_desc, X_qp], "DFT_ADME": [X_dft, X_qp], "Desc_DFT": [X_desc, X_dft],
    #               "Desc_ADME_DFT": [X_desc, X_qp, X_dft]}
                 
    #tests_dict_naphtho = {"Desc_ADME": [X_desc, X_qp]}

    tests_dict = {"Morgan": [X_morgan], "Desc": [X_desc], "ADME": [X_qp], "DFT": [X_dft],
                  "Desc_ADME": [X_desc, X_qp], "DFT_ADME": [X_dft, X_qp], "Desc_DFT": [X_desc, X_dft],
                  "Desc_ADME_DFT": [X_desc, X_qp, X_dft]}
    
    graphs = {"Morgan": False, "Desc": False, "ADME": False, "DFT": False,
                  "Desc_ADME": True, "DFT_ADME": False, "Desc_DFT": False,
                  "Desc_ADME_DFT": True}

    expr_test_dict = {"Morgan": False, "Desc": True, "ADME": True, "DFT": True,
                  "Desc_ADME": True, "DFT_ADME": True, "Desc_DFT": True,
                  "Desc_ADME_DFT": True}
    
    test = list(set(list(range(rows))) - set(train))
    
    reduce_list(train, remove_entry)
    reduce_list(test, remove_entry)
    y=df_redox['Experimental Redox'].values
    ytrain = y[train]
    ytest = y[test]    
    for k,v in tests_dict.items():
        X = v[0]
        for item in v[1:]:
            X = np.concatenate((X, item), axis=1)
            
        Xtrain = X[train]
        Xtest = X[test]

        if expr_test and expr_test_dict[k]:
            v_expr = tests_dict_expr[k]
            X_expr = v_expr[0]
            for item in v_expr[1:]:
                X_expr = np.concatenate((X_expr, item), axis=1)
        else:
            X_expr = []
            #y_expr = []
            
        ypred, ypred_expr = calib_gpr(Xtrain, Xtest, ytrain, ytest, X_expr, y_expr, expr_test, cvFlag=False)
        if not pred_only:
            r2 = metrics.r2_score(ytest, ypred)
            print("R2 for {} = {}".format(k, r2))
            if graphs[k]:
                file = k + '_' + dataset + '_' + 'scatter.png'
                label_x = "Predicted Redox"
                label_y = "Experimental Redox"
                make_scatter_plot(ypred, ytest, r2, label_x, label_y, file)
            
        if expr_test and expr_test_dict[k]:
            file = 'expr' + k + '_' + dataset + '_' + dataset_test + '_' + 'values.csv'
            mol_l = df_redox_expr['LHS Compound'].tolist()
            expr_df = pd.DataFrame({'Molecule': mol_l, 'Redox': ypred_expr+shift_expr})
            expr_df.to_csv(file, index=False)
            if not pred_only:
                r2_expr = metrics.r2_score(y_expr, ypred_expr+shift_expr)
                print("R2 for expr {} = {}".format(k, r2_expr))
                file = 'expr' + k + '_' + dataset + '_' + 'scatter.png'            
                label_x = "Predicted Redox"
                label_y = "Experimental Redox"
                make_scatter_plot(ypred_expr+shift_expr, y_expr, r2_expr, label_x, label_y, file, True)
            
