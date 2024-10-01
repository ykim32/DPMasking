"""
* Dual-process masking 
  - utility functions
  Date: 6/5/2024
"""

import pandas as pd
import numpy as np
import random
import os
import gc
import torch
import shutil 
import time
import datetime
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from sklearn.model_selection import KFold


def clearMem(path):
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(path) 
    
def printExpTime(startTime):
    totLearnTime = (time.time() - startTime)/60 
    print("Total Time: {:.1f} min ({:.1f} hours) - {}".format(totLearnTime, totLearnTime/60, datetime.datetime.now()))

    
def kfold_seperateTrajectories(df, n_splits):
    foldIDnum = int(np.round(len(df.sid.unique())/n_splits, 0))
    for fold_no in range(n_splits):
        if fold_no == n_splits-1:
            foldIDs = df.sid.unique()[fold_no*foldIDnum:]
        else:
            foldIDs = df.sid.unique()[fold_no*foldIDnum: (fold_no+1)*foldIDnum]
        yield df[~df.sid.isin(foldIDs)].index, df[df.sid.isin(foldIDs)].index

def val_seperateTrajectories(traindf, val_size, groupID='sid'):
    trainIDs = traindf[groupID].unique().tolist()
    valIDs = random.sample(trainIDs, k=int(val_size*len(trainIDs)))
    return traindf[~traindf[groupID].isin(valIDs)].index, traindf[traindf[groupID].isin(valIDs)].index


def getOneHotLabels(df, labelNames, targetLabels, prefix='y_'):
    y_labels = [prefix+l.split(' ')[0] for l in labelNames]
    for i in range(len(labelNames)):
        df[y_labels[i]] = 0
        df.loc[df[targetLabels].str.contains(labelNames[i]),y_labels[i]] = 1
    return df, y_labels


def initFile(filePath, columns=[]):
    if os.path.exists(filePath):
        return pd.read_csv(filePath, header=0)
    else:
        return pd.DataFrame(columns = columns)


def checkEarlyStop(df, warmup, patience, t_metric='acc'):
    vdf = df.copy(deep=True)
    if t_metric == 'f1_acc':
        vdf[t_metric] = (vdf['acc'] + vdf['f1'])/2
        
    vdf = vdf[vdf.epoch > warmup]
    #print(vdf)
    if patience == 1:
        earlyStop = (vdf.shift(-1)[t_metric] -vdf[t_metric] <= 0 )
    elif patience == 2:
        earlyStop = ((vdf.shift(-1)[t_metric] -vdf[t_metric] <= 0 ) & ( vdf.shift(-2)[t_metric]-vdf[t_metric]  <= 0 ))        
    else: #  patience == 3:
        earlyStop = ((vdf.shift(-1)[t_metric] -vdf[t_metric] <= 0 ) & ( vdf.shift(-2)[t_metric]-vdf[t_metric]  <= 0 )
                     & (vdf.shift(-3)[t_metric]-vdf[t_metric]  <= 0 ))
        
    if earlyStop.any():
        esEpoch = int(vdf[earlyStop].head(1).epoch.values[0]) # early stop epoch
        if esEpoch > warmup:
            print(" !!!! Early stop {}: {} (patience: {})".format(earlyStop.any(), esEpoch, patience))  
            earlyStopped = True
        else:
            print(" !!!! Early stop {} : {} due to warmup (patience: {})".format(earlyStop.any(), esEpoch, patience))  
            if len(vdf)>0: esEpoch = int(vdf.tail(1).epoch.values[0])
            else: esEpoch = 0
            earlyStopped = False
    else:
        if len(vdf)>0: esEpoch = int(vdf.tail(1).epoch.values[0])
        else: esEpoch = 0
        earlyStopped = False

    return earlyStopped, esEpoch 

    
def getEpoch(s):
    pos = s.find('epoch')
    s = s[pos:].split('/')[0]
    epoch = int(s[len('epoch'):])
    return epoch


def evaluate_noCV(df, labelNames, lr, batch, trainFile, testFile, path, allres_file, curEpoch):
    y_labels = ['y_' + l for l in labelNames]
    ypred_labels = [l.replace('y', 'ypred') for l in y_labels]
    f1_labels = ['f1_' + l for l in labelNames]
    mcc_labels = ['mcc_' + l for l in labelNames]
    evdf = pd.DataFrame(columns = f1_labels + mcc_labels)
    
    df['label'] = df.label.str.strip()     

    metric_f1, metric_mcc = [], []
    
    # Multi-labels Multi-classification
    df, _ = getOneHotLabels(df, labelNames, targetLabels= 'label', prefix='y_')
    df, ypred_labels = getOneHotLabels(df, labelNames, targetLabels= 'pred', prefix='ypred_')

    macro_f1 = f1_score(df[y_labels].values.tolist(), df[ypred_labels].values.tolist(), average='macro')
    macro_mcc = matthews_corrcoef(df[y_labels].values.tolist(), df[ypred_labels].values.tolist())
    df.to_csv(path+'resdf.csv', index=False)  
    evdf.loc[len(evdf)] = metric_f1 + metric_mcc    
    
    sigNum = 4
    avgOverallF1 = np.round(np.mean(macro_f1), sigNum) #np.round(evdf[f1_labels].mean().mean(), sigNum)
    avgOverallMCC = np.round(np.mean(macro_mcc), sigNum) #np.round(matthews_corrcoef(df['label'], df['pred']), sigNum)
    acc = np.round(accuracy_score(df['label'], df['pred']), sigNum)
    print("\n* Overall Avg-F1: {:.3f}\tAvg-MCC: {:.3f}\tACC: {:.3f}".format(avgOverallF1, avgOverallMCC, acc))    
    
    evdf[evdf.columns[1:]] = evdf[evdf.columns[1:]].round(3)
    evdf.to_csv(path+'eval.csv', index=False) 
    
    ## all methods' results -------------------
    ares = initFile(allres_file, columns = ['model', 'lr', 'batch', 'trainFile', 'testFile'] + ['macro_f1', 'macro_mcc', 'acc', 'epoch'])

    modelPath = path[28:]    
    if curEpoch > -1: epoch = curEpoch
    elif 'epoch' in modelPath: epoch = getEpoch(modelPath)
    else: epoch = 0
    ares.loc[len(ares)] = [modelPath, lr, batch, trainFile, testFile] + [avgOverallF1, avgOverallMCC, acc, epoch]
    cols = ['model', 'lr', 'batch', 'testFile', 'macro_f1', 'macro_mcc', 'acc', 'epoch'] 
    
    print("** Recent results:\n", ares.tail(20)[cols])
    
    baseF1, baseAcc = ares.macro_f1[0]*100, ares.acc[0]*100
    
    #testRes = ares[ares.testFile.str.contains('test')]
    if ares.testFile.str.contains('u_val').any():
        testRes = ares[ares.testFile.str.contains('u_val')]
    elif ares.testFile.str.contains('val').any():
        testRes = ares[ares.testFile.str.contains('val')]
    else: testRes = []
    
    if len(testRes) > 0:
        maxF1, maxAcc = testRes.macro_f1.max()*100, testRes.acc.max()*100
        bestAcc = testRes.loc[testRes.acc.idxmax()]    
        bestF1 = testRes.loc[testRes.macro_f1.idxmax()]

        print("Base Acc: {:.1f}%\tMax ACC: {:.1f}% (+{:.1f}%) - epoch: {} (b={}, lr={})".format(baseAcc, maxAcc, (maxAcc-baseAcc),  
                                bestAcc.epoch, bestAcc.batch, bestAcc.lr))    
        print("Base F1: {:.1f}%\tMax F1: {:.1f}% (+{:.1f}%) - epoch: {} (b={}, lr={})".format(baseF1, maxF1, (maxF1-baseF1), 
                                bestF1.epoch, bestF1.batch, bestF1.lr))
    
    if ares.testFile.str.contains('test').any():
        testRes = ares[ares.testFile.str.contains('test')] 
        maxF1, maxAcc = testRes.macro_f1.max()*100, testRes.acc.max()*100
        bestAcc = testRes.loc[testRes.acc.idxmax()]    
        bestF1 = testRes.loc[testRes.macro_f1.idxmax()]
        print("Test best ACC {:.1f}% - epoch: {} (b={}, lr={})".format(maxAcc, bestAcc.epoch,  bestAcc.batch, bestAcc.lr ))
        print("Test best F1 {:.1f}% - epoch: {} (b={}, lr={})".format(maxF1, bestF1.epoch, bestF1.batch, bestF1.lr))
    ares.to_csv(allres_file, index=False)
    
    return maxAcc, acc, avgOverallF1, avgOverallMCC


