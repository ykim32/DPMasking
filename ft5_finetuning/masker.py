"""
* Dual-process masking 
  - Masker class 
 Date: 6/5/2024
"""

import argparse
import time
import datetime
import os
import pandas as pd
import numpy as np
import shutil
import random
import string
import gc
import torch
from sklearn.model_selection import train_test_split
from transformers import set_seed
from collections import deque

import train_lm as lm
import util_mask as ut


def getMainArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-val_size', type=float, default=0.2) 
    parser.add_argument('-lr', type=float, default=0.0003)     
    parser.add_argument('-batch', type=int, default=4)
    parser.add_argument('-startEpoch', type=int, default=0)
    parser.add_argument('-endEpoch', type=int, default=10)    
    parser.add_argument('-debug', type=int, default=0)
    args = parser.parse_args()  
    print(args)
    return args  


class Args():
    speaker = 'teacher'
    maxlen = 1024
    maxSeqLen = 1
    
    def __init__(self, pathKey, mntPath, pretrain, targetX, targetY, targetTrain, targetVal, targetTest, lr, batch, test_batch, epoch,
                 keyword, tMask, maskMode, saveLimit, groupID, curEpoch=-1):
        self.pathKey = pathKey
        self.mntPath = mntPath
        self.pretrain = pretrain
        self.targetX = targetX
        self.targetY = targetY
        self.targetTrain = targetTrain
        self.targetVal = targetVal
        self.targetTest = targetTest
        self.epoch = epoch
        self.batch = batch
        self.test_batch = test_batch
        self.lr = lr
        self.keyword = keyword
        self.tMask = tMask
        self.maskMode = maskMode
        self.curEpoch = curEpoch
        self.saveLimit= saveLimit
        self.groupID = groupID

def setPath(path):
    if not os.path.exists(path): os.makedirs(path) 


class Masker():
    
    deltaT = 1

    cInFeat = 'clsInFeat'
    maskFeat = 'masked_word'
    
    def __init__(self, mainArgs):
        self.maskMode = mainArgs.maskMode
        self.maskPercent = mainArgs.maskPercent
        self.groupID = mainArgs.groupID 
        self.saveLimit= mainArgs.saveLimit
        self.patience = mainArgs.patience
        self.maskTest = mainArgs.maskTest
        self.dataPercent = mainArgs.dataPercent

        self.speaker = mainArgs.speaker
        self.truncRightAlign = mainArgs.truncRightAlign
        self.tMask = mainArgs.tMask
        self.mInFeat = mainArgs.inputFeat
        
        self.masker_cols = [self.groupID, self.mInFeat, 'labels', 'tokenNum', self.maskFeat, 'idx'] 
        self.cls_cols = [self.groupID, self.cInFeat, 'labels', 'tokenNum', self.maskFeat, 'idx']   

        self.masker_train = None
        self.i_train = None
        self.i_val = None
        self.i_test = None
        
        self.mntPath = mainArgs.mntPath
        self.pathKey = mainArgs.pathKey

        self.maxTokenNum = mainArgs.maxTokenNum
        
        self.smem = None # same size datafram to the trainVal 
        self.success_mask_idx = []
        self.fail_mask_idx = []
        
        self.val_size = mainArgs.val_size
        self.batch = mainArgs.batch
        self.test_batch = mainArgs.test_batch
        self.lr = mainArgs.lr
        self.startEpoch = mainArgs.startEpoch
        self.endEpoch = mainArgs.endEpoch
        self.trainEpoch = mainArgs.trainEpoch
        self.randSeed = mainArgs.randSeed
        self.keyword = 'ds{}_dp{}_mT{}_tE{}_{}_{}'.format(mainArgs.dataPercent, mainArgs.maskPercent, mainArgs.maskTest, mainArgs.trainEpoch, mainArgs.keyword, mainArgs.randSeed)
        self.hyperParam = 'v{}b{}l{}t{}_{}'.format(int(self.val_size*10), self.batch, int(np.ceil(self.lr*10000)), self.maxTokenNum, self.keyword)
        
        if self.maskMode == 1  : self.modelName = 'rm{}_{}'.format(self.maskPercent, self.hyperParam)
        elif self.maskMode == 2: self.modelName = 'dp{}_{}'.format(self.maskPercent, self.hyperParam)
        else: self.modelName = 'base{}_{}'.format(self.keyword, self.hyperParam)
        
        
        self.mainPath = '{}data/{}mask/{}/'.format(self.mntPath, self.pathKey, self.modelName)
        setPath(self.mainPath)
        self.debug = mainArgs.debug


    def saveHyper(self, margs):
        with open(self.mainPath+'/hyperparameter.txt', 'w') as f:
            margs_dict = vars(margs)
            f.write(str(margs_dict))
            
    def trainModel(self, margs, targetTrain, targetVal, targetTest, curEpoch, batch, lr, targetX, pretrain=None):
        args = Args(pathKey=margs.pathKey, mntPath=margs.mntPath, pretrain=pretrain, targetX=targetX, targetY='labels', 
                    targetTrain=targetTrain, targetVal=targetVal, targetTest=targetTest, lr=lr, batch=batch, test_batch=margs.test_batch, epoch=margs.trainEpoch, 
                    keyword=margs.keyword, tMask=margs.tMask, maskMode=margs.maskMode, curEpoch=curEpoch, saveLimit=self.saveLimit, groupID=self.groupID)

        _, acc, f1, mcc, classifierPath, resFilePath = lm.main(args)
        return acc, f1, mcc, classifierPath, resFilePath
            
        
    # randomly mask a word within a given token length 
    def random_select(self, x, tokenNum):
        ### tokenNum decreases by removing self.tMask
        
        x = "".join([i for i in str(x) if i not in string.punctuation]) # exclude punctuation from mask   
        splited_x = x.split()
        tokenNum = len(splited_x)
        
        targetSent = x.replace("#", "@")
        sents = targetSent.split("@")
        targetSent = sents[-1]
        targetTokenNum = len(targetSent.split())
        
        if tokenNum > 3 and targetTokenNum > 3: # include punctuation  # add targetTokenNum (11/11)
            randomNum = random.randint(0, tokenNum-1) # exclude '#' or '@'
        else:
            return self.tMask
       
        target = splited_x[randomNum] + ' ' 
        return target

    def removeToken(self, s, tokens):
        removed = [t for t in str(s).split() if t not in str(tokens).split()]
        return ' '.join(removed)

    def removeMaskTokenWhenHavingValidTokens(self, s, tokens):
        if len(str(s).split()) > 1:        
            removed = [t for t in str(s).split() if t not in str(tokens).split()]
            return ' '.join(removed)
        else:
            return s
    

    # def copy_mask(self, a):
    #     splitted = str(a).split()
    #     if len(splitted)>0: return splitted[0]    
    #     else: return self.tMask
    
    
    def mask(self, df, targetFeat='Sentence', random=0, targetSent = 'masked_sent'):
        df[targetSent] = df[targetFeat].str.replace("Classify: ", "").str.replace("Mask: ", "")        
        if random: 
            df['mask_candidates'] = df[targetSent].str.replace('!', '').str.replace('?', '').str.replace("\.", "").str.replace("\,", "").str.replace(self.tMask, "", regex=False) #  11/11          
            df['tokenNum_cand'] = df['mask_candidates'].str.split().str.len()
            df['single_masked_word'] = df.apply(lambda x: self.random_select(x['mask_candidates'], x['tokenNum_cand']), axis=1).str.replace(' ', '') 

        df[targetSent] = df.apply(lambda x: self.replaceToken(x[targetSent], x['single_masked_word']), axis=1)    
        df[targetSent] = df[targetSent].apply(lambda x: self.removeDupSpace(x))
        return df

    # =======================================
    # Preprocessing
    def punctuation_with_space(self, text):
        return "".join(' '+c if c in string.punctuation.replace("'","") else c for c in str(text))

    def remove_punctuation(self, text):
        if self.speaker != '': clean = "".join([i for i in text if i not in string.punctuation.replace('@','').replace('#','')])                    
        else: clean="".join([i for i in str(text) if i not in string.punctuation])
        return clean  

    def removeDupSpace(self, x):
        return " ".join(str(x).split())
    
    def replaceToken(self, s, tokens): 
        splitted = str(s).split()
        tokenList = str(tokens).split()
        for i in range(len(splitted)):
            if splitted[i] in tokenList: splitted[i] = self.tMask
        return ' '.join(splitted)    
    
    def truncateTokens(self, df, maxTokenNum, targetFeat):
        df = df[df[targetFeat]!='']
        df.reset_index(drop=True, inplace=True)

        if maxTokenNum > 0 :
            df['tokenNum'] = df[targetFeat].str.split().str.len()
            tmp = df[df.tokenNum>maxTokenNum]
            idx = tmp.index
            tmp.reset_index(drop=True, inplace=True)

            if self.truncRightAlign:
                for i in range(len(tmp)):
                    df.loc[idx[i], targetFeat] = ' '.join(tmp.loc[i][targetFeat].split()[-maxTokenNum:])  # right align
            else: # leftAlign
                for i in range(len(tmp)):
                    df.loc[idx[i], targetFeat] = ' '.join(tmp.loc[i][targetFeat].split()[:maxTokenNum])  # left align            

        df['tokenNum'] = df[targetFeat].str.replace("@ ", "").str.replace("# ","").str.split().str.len()    
        df = df[df.tokenNum > 0]
        df.reset_index(drop=True, inplace=True)
        print(" - token num: mean {:.1f}, max {}".format(df['tokenNum'].mean(), df['tokenNum'].max()))
        return df
    
    def handlePunctuation_addPrefix(self, df, removePunct, inputFeat):
        if removePunct: 
            df[inputFeat] = df[inputFeat].str.replace("Classify: ", '')
            df[inputFeat] = df[inputFeat].apply(lambda x: self.remove_punctuation(x))
            df[inputFeat] = "Classify: " + df[inputFeat]
        else:
            df[inputFeat] = df[inputFeat].apply(lambda x: self.punctuation_with_space(x).replace("Classify : ", "Classify:"))
        return df
    
    
    def loadPreprocData(self, origin, dataName, inputFeat, removePunct):        
        dataPath = '{}data/{}msl1/'.format(self.mntPath, self.pathKey)
        if not os.path.exists(dataPath): os.makedirs(dataPath)
        targetData = '{}{}.csv'.format(dataPath, dataName)   
        shutil.copy(origin, targetData)     
        
        df = pd.read_csv(targetData, header=0)
        df = df[df[inputFeat]!=' ']
        df = df[df[inputFeat].notnull()]
        
        # Truncate data with the given dataPercent --------
        orgNum = len(df)
        if (self.dataPercent < 1) and ('test' not in dataName):
            df = df.loc[:int(len(df)*self.dataPercent)]   
        print("{} Data percent: {}% ({} / {})".format(dataName, self.dataPercent*100, len(df), orgNum))
         
        if self.debug:
            if dataName =='test' or dataName == 'val': df = pd.concat([df.head(5), df.tail(5)], axis=0)            
            else:                 df = pd.concat([df.head(50), df.tail(50)], axis=0)            

        df.reset_index(drop=True, inplace=True)
       
        df[inputFeat] = df[inputFeat].str.replace('Classify: ', '')
        df = self.handlePunctuation_addPrefix(df, removePunct, inputFeat)  
        
        df[inputFeat] = df[inputFeat].apply(lambda x: self.removeDupSpace(x))    
        df = self.truncateTokens(df, self.maxTokenNum, inputFeat) 
        return df, targetData    

    def loadPreprocData_DP(self, inputFeat, dataName, em_epoch=-1, removePunct=1):
        targetData = '{}data/{}{}.csv'.format(self.mntPath, self.pathKey, dataName)        
        df = pd.read_csv(targetData, header=0)
        
        df = df[df[inputFeat]!=' ']
        df = df[df[inputFeat].notnull()]
        
        orgNum = len(df)
        if (self.dataPercent < 1) and ('test' not in dataName):
            df = df.loc[:int(len(df)*self.dataPercent)]   
        print("Data percent: {}% ({} / {})".format(self.dataPercent*100, len(df), orgNum))
        df.reset_index(drop=True, inplace=True)
        
        if self.debug:
            if dataName =='test' or dataName == 'val': df = pd.concat([df.head(16), df.tail(16)], axis=0)            
            else:                 df = pd.concat([df.head(64), df.tail(64)], axis=0) 
                    
        df.reset_index(drop=True, inplace=True)
        df = self.handlePunctuation_addPrefix(df, removePunct, inputFeat)
                
        df[inputFeat] = df[inputFeat].apply(lambda x: self.removeDupSpace(x))    
        # if randomShuffle: df = df.sample(frac=1)
        df = self.truncateTokens(df, self.maxTokenNum, inputFeat)        
        return df
    
    
    # Preprocessing for No-masking baseline approach
    def loadPreprocBaseData(self, inputFeat, dataName, removePunct=0, randomShuffle=0):
        origin = '{}data/{}{}.csv'.format(self.mntPath, self.pathKey, dataName)
        df, targetData = self.loadPreprocData(origin, dataName, inputFeat, removePunct)
        
        df[self.cInFeat] = "Classify: " + df[inputFeat]
        if randomShuffle: df = df.sample(frac=1)
        
        #df.to_csv('{}data/{}msl1/{}.csv'.format(self.mntPath, self.pathKey, dataName), index=False)
        df.to_csv(targetData, index=False)
        if self.debug: print(df.head(2))
        return df
    
    # Preprocessing for Dynamic Randome Masking
    def loadPreprocRandomMaskData(self, inputFeat, dataName, removePunct=1, randomMask=0, randomShuffle=0):
        
        origin = '{}data/{}/{}.csv'.format(self.mntPath, "/".join(self.pathKey.split('/')[:-2])+'/', dataName)
        df, targetData = self.loadPreprocData(origin, dataName, inputFeat, removePunct)

        cols = [self.groupID, self.cInFeat, 'labels']
        
        if randomMask:
            self.multi_mask_random_percent(df, inputFeat, percent=self.maskPercent)  # output: masked_sent   
            df[self.cInFeat] = "Classify: " + df['masked_sent']
            df = df[cols]
            if self.debug: print("Train: ", df[self.cInFeat].head(3))
        else:
            df[self.cInFeat] = "Classify: " + df[self.mInFeat]
            df = df[cols]
            if self.debug: print("Test: ", df[self.cInFeat].head(3))
            
        if randomShuffle: df = df.sample(frac=1)            
        
        df.to_csv(targetData, index=False)
        if self.debug: print(df.head(2))        
        return df
        
        
    def randomMask_allSentence(self, df, inputFeat):
        if self.maskMode == 2: self.multi_mask_random_percent(df, self.mInFeat, percent=self.maskPercent) 
        elif self.maskMode ==3: self.multi_mask_random_count(df, self.mInFeat, repeat=1)
        else:
            df = self.mask(df, inputFeat, random=1) # outputFeat: "masked_sent"
            df = self.handleNullTokens(df, 'masked_word')
        print("init randomMask: avg Mask Size {:.2f} (instances: {})".format(df['masked_word'].str.split().str.len().mean(), len(df)))
        # df.to_csv('data/{}mask_{}.csv'.format( self.pathKey, dataName), index=False)
        return df

    def moveUnselectedToVal(self, trainIdx, valIdx):
        selected = self.smem.loc[valIdx]
        selected = selected[selected.selected==1].idx.values.tolist()
        selectedNum = len(selected)
        unselected = self.smem.loc[trainIdx]
        unselected = unselected[unselected.selected!=1].idx.values.tolist()
        print("  * Unselected num from trainIdx: {} / {}".format(len(unselected), len(trainIdx)))
        if len(unselected) > 0:        
            valIdx = [i for i in valIdx if i not in selected] + unselected[:selectedNum]
            trainIdx =[i for i in trainIdx if i not in unselected] + unselected[selectedNum:] + selected
            print("  * Swap num in valIdx: {} / {}".format(selectedNum, len(valIdx))) 
        else:
            failed = self.smem.loc[valIdx]
            failed = failed[failed.correct!=1].idx.values.tolist()
            failedNum = len(failed)
            print("  * Failed num: {} / {}".format(failedNum, len(valIdx)))
            succeeded = self.smem.loc[trainIdx]
            succeeded = succeeded[succeeded.correct!=1].idx.values.tolist()
            if len(succeeded) > 0: 
                valIdx = [i for i in valIdx if i not in failed] + succeeded[:failedNum]
                trainIdx =[i for i in trainIdx if i not in succeeded] + succeeded[failedNum:] + failed
                print("  * Swap num: {}".format(failedNum))   
        return trainIdx, valIdx


    # 
    def initDataRandomMask(self, inputFeat, splitMode, em_epoch):
        self.i_train = self.randomMask_allSentence(self.i_train, inputFeat)
        self.i_val = self.randomMask_allSentence(self.i_val, inputFeat)        
        self.i_test = self.randomMask_allSentence(self.i_test , inputFeat)
        self.i_train['idx'] = self.i_train.index
        self.i_val['idx'] = self.i_val.index        
        self.i_test['idx'] = self.i_test.index

        # data split for Concisous learning 
        if splitMode=='byGroup':
            r_trainIdx, r_valIdx = ut.val_seperateTrajectories(self.i_train, self.val_size, self.groupID)
        else:
            r_trainIdx, r_valIdx = train_test_split(self.i_train.index.tolist(), test_size=self.val_size, shuffle=True, random_state=em_epoch)
                                             #   stratify=self.i_train.labels, random_state=em_epoch)
            
        # Move unselected instances to the current validation set, and move the same number of instances to  the current training set.
        if em_epoch>1: r_trainIdx, r_valIdx = self.moveUnselectedToVal(r_trainIdx, r_valIdx)        
                        
        self.r_train = self.i_train.loc[r_trainIdx]
        self.r_val = self.i_train.loc[r_valIdx]  
                        
        self.r_train.reset_index(drop=True, inplace=True)
        self.r_val.reset_index(drop=True, inplace=True)

    def initDataRandomMask_R(self, inputFeat, splitMode, em_epoch):
        # data split for Concisous learning 
        if splitMode=='byGroup':
            r_trainIdx, r_valIdx = ut.val_seperateTrajectories(self.i_train, self.val_size, self.groupID)
        else:
            r_trainIdx, r_valIdx = train_test_split(self.i_train.index.tolist(), test_size=self.val_size, shuffle=True, random_state=em_epoch)
                                             #   stratify=self.i_train.labels, random_state=em_epoch)
            
        if em_epoch>1: r_trainIdx, r_valIdx = self.moveUnselectedToVal(r_trainIdx, r_valIdx)        
                        
        self.r_train = self.i_train.loc[r_trainIdx]
        self.r_val = self.i_train.loc[r_valIdx]  
                        
        self.r_train.reset_index(drop=True, inplace=True)
        self.r_val.reset_index(drop=True, inplace=True)        

    def initSmem(self):
        print("* Init S-mem")
        if self.startEpoch <= 1:  
            self.smem = self.i_train.copy(deep=True)
            self.smem['correct'] = np.nan
            self.smem['valid_correct'] = np.nan 
            self.smem['pred_mask'] = str("")
        else: # load the previous smem
            path = '{}/epoch{}/masker/msl1/'.format(self.mainPath, self.startEpoch)
            self.smem = pd.read_csv(path+'/smem_all.csv', header=0) # Load success memory with empty mask
        if self.debug:
            print(self.smem[['idx', self.mInFeat, 'masked_word', 'masked_sent', 'correct', 'valid_correct', 'pred_mask']].head(3))


    def removeDuplicate(self, a):
        return " ".join(set(str(a).split()))
    
    
    # max number of token
    def multi_mask_random_count(self, df, inputFeat, repeat):  
        df['multi_mask'] = str("")
        for i in range(repeat):
            if i ==0:
                df['masked_sent'] = df[inputFeat]
            df = self.mask(df, targetFeat='masked_sent', random=1)   # new random mask
            df['multi_mask'] += ' '*int(bool(i)) + df.single_masked_word
            
        df['multi_mask'] = df.multi_mask.apply(lambda x: self.removeDuplicate(x)) # remove duplicated masks
        df['multi_mask'] = df['multi_mask'].apply(lambda x: self.removeMaskTokenWhenHavingValidTokens(x, self.tMask))
        #df['multi_mask'] = df.multi_mask.str.replace(self.tMask+" ", "").replace(" "+self.tMask, "")
        # ============
        df['masked_word'] = df['multi_mask']
        # ============        
        df.drop(columns = ['tokenNum_cand'], inplace=True)
        
   
    def multi_mask_random_percent(self, df, inputFeat, percent=0.15):  
        df['multi_mask'] = str("")
        df['maxMaskNum'] = (df['tokenNum'] * percent).astype('int')
        
        df.loc[(df.tokenNum > 1)&(df.maxMaskNum==0), 'maxMaskNum'] = 1
        
        repeat = int(df['maxMaskNum'].fillna(0).max())
        
        for i in range(repeat):
            if i ==0:
                df['masked_sent'] = df[inputFeat]
            target = df[df.maxMaskNum > i] 
            target = self.mask(target, targetFeat='masked_sent', random=1)   # new random mask
            df.loc[target.index, 'multi_mask']  += ' '*int(bool(i)) + target.single_masked_word
            df.loc[target.index, 'masked_sent']  = target.masked_sent
            
        df['multi_mask'] = df.multi_mask.apply(lambda x: self.removeDuplicate(x)) # remove duplicated masks
        df['multi_mask'] = df['multi_mask'].apply(lambda x: self.removeMaskTokenWhenHavingValidTokens(x, self.tMask))
        # ============
        df['masked_word'] = df['multi_mask']
        # ============   
        if 'tokenNum_cand' in df.columns:
            df.drop(columns = ['tokenNum_cand'], inplace=True)        


    # For the current random initialization of masking in trainIdx, we retrieve and utilize a successful masking memory from S-MEM.
    def preproc_R_Masker(self, epoch, maskFeat = 'masked_word', sampleRate=1, sampleIdx=0): 
        # By incorporating a new random mask into the existing successful memory mask, the data for masker_train in this epoch is finalized.
        # All masked sentences from S-MEM are copied over, and we start with the candidates excluding the existing successful masks.
        if epoch == 1: self.smem['masked_sent'] = self.smem[self.mInFeat] # initialize S-MEM when starting the training
            
        # 1) Copy S-MEM (including succesful masks) for the training data of Masker (masker_train) 
        self.masker_train = self.smem.loc[self.r_train.idx].copy(deep=True)   
        
        # 2) Remove failed masks, while retaining the existing masked_sent, 
        #     in order to reduce instances of immediately selecting the wrong mask and facilitate the selection of new masks.
        self.masker_train.loc[self.masker_train.correct!=1, 'masked_word'] = str("")
        
        # 3) Add new (valid) random masks for exploration.
        self.masker_train = self.mask(self.masker_train, targetFeat='masked_sent', random=1)   # new random mask
        
        con = self.masker_train['single_masked_word'] != self.tMask
        self.masker_train.loc[con, 'masked_word'] +=  " " + self.masker_train.loc[con, 'single_masked_word']
        self.masker_train[maskFeat] = self.masker_train[maskFeat].apply(lambda x: self.removeMaskTokenWhenHavingValidTokens(x, self.tMask))
        
        numValidMask =  len(self.masker_train[self.masker_train[maskFeat]!=self.tMask])
        print(" * Init valid random mask rate: {:.2f}".format(numValidMask/len(self.masker_train)))

        # 4) Generate and save the training / test data for Masker
        
        # masker_train_sample = self.masker_train.sample(frac=sampleRate)

        if sampleRate < 1: 
            sampleNum = int(len(self.masker_train) * sampleRate)
            self.masker_train.reset_index(drop=True, inplace=True)
            masker_train_sample = self.masker_train.loc[sampleIdx*sampleNum:(sampleIdx+1)*sampleNum]
        else:  masker_train_sample = self.masker_train
        
        
        self.masker_train = self.saveMaskerData(masker_train_sample, maskFeat, "masker_train", epoch, debug=1)            
        self.smem = self.saveMaskerData(self.smem, maskFeat, "r_masker_smem", epoch)  # update S-MEM with new r_pred_mask


    # Generate and save the data for training Masker (both R-Masker or I-Masker)
    def saveMaskerData(self, df, maskFeat, dataName, epoch, debug=0):
        path = '{}/epoch{}/masker/msl1/'.format(self.mainPath, epoch)
        setPath(path)
        df = self.handleNullTokens(df, maskFeat)        
        mdf = df.copy(deep=True)
        mdf[self.mInFeat] = "Mask: " + mdf[self.mInFeat].str.replace("Mask: ", "")  
        mdf[self.masker_cols].to_csv(path+'{}.csv'.format(dataName), index=False)
        print("{} size: {}".format(dataName, len(mdf))) 
        if self.debug:
            print("{}".format(mdf[self.masker_cols].head(3)))
        #if self.debug: print("{} size: {}\n{}".format(dataName, len(mdf), mdf[self.masker_cols].head(3))) 
        return df
        
    def getMaskSize(self, df, maskSizeFeat='maskSize', maskFeat='multi_mask'):
        df[maskSizeFeat] = df[maskFeat].str.split().str.len()
        df.loc[df[maskFeat]==self.tMask, maskSizeFeat] = 0
        if self.debug: print("avg. maskSize {:.1f} , num of empty masks: {}".format(df[maskSizeFeat].mean(), len(df[df[maskSizeFeat]==0])))
        return df

    def handleNullTokens(self, df, feat):
        df[feat] = df[feat].fillna(self.tMask)          
        df[feat] = df[feat].astype("str")
        df.loc[df[feat]=='nan', feat] = self.tMask
        df[feat] = df[feat].apply(lambda x: self.removeDuplicate(x))
        df[feat] = df[feat].apply(lambda x: self.removeMaskTokenWhenHavingValidTokens(x, self.tMask))
        #df[feat] = df[feat].str.replace(self.tMask+" ", "").str.replace(" "+self.tMask, "")
        df[feat] = df[feat].apply(lambda x: self.removeDupSpace(x))        
        return df

    def validMask(self, s, mask): 
        s, mask = str(s), str(mask)
        s = "".join([i for i in s if i not in string.punctuation])
        splitted = s.split()
        if len(splitted) > 2:
            con = (mask in splitted) 
            return mask if con else str("")   
        else:
            return str("")
    
    def getValidMask(self, df, epoch, dataName, maskFeat):
        
        if self.debug: print(" # Check mask validity -----------------------------")
        df['valid_mask'] = str("")
        
        for i in range(df[maskFeat].str.split().str.len().max()+1):
            df['tmask'] = df[maskFeat].str.split().str[i]
            df['valid_mask'] += " "*int(bool(i)) + df.apply(lambda x: self.validMask(x[self.mInFeat], x['tmask']), axis=1)
        
        df['valid_mask'] = df['valid_mask'].apply(lambda x: self.removeDupSpace(x))
        df.loc[df['valid_mask']=='', 'valid_mask'] = self.tMask
        df = self.handleNullTokens(df, 'valid_mask')
        df.drop(columns = ['tmask'], inplace=True)

        df = self.getMaskSize(df, maskSizeFeat='valid_maskSize', maskFeat='valid_mask')
        invalid = df.loc[df.maskSize != df.valid_maskSize].copy(deep=True)
        invalid.to_csv(self.mainPath + 'epoch{}/invalidMask_{}.csv'.format(epoch, dataName), index=False)
        
        invalidRate =  np.round(len(invalid)/len(df), 4)
        predMaskSize = np.round(df.maskSize.mean(), 3)        
        validMaskSize = np.round(df.valid_maskSize.mean(), 3)
        maskSizeRatio = np.round((validMaskSize/predMaskSize), 3)
        avgMaskRatePerInput = np.round((df.valid_maskSize/df.tokenNum).mean()*100, 2)
        if self.debug: 
            print("  - predicted mask size: {:.3f}, valid mask size: {:.3f} (ratio: {:.3f})".format(predMaskSize, validMaskSize, maskSizeRatio))
            print("  - avgMaskRatePerInput: {:.2f}%".format(avgMaskRatePerInput))
            print("  - Invalid mask num: {}/{}, rate: {}".format(len(invalid), len(df), invalidRate))
        
        maskLogFile = self.mainPath + 'maskLog.csv'
        if os.path.exists(maskLogFile):
            log = pd.read_csv(maskLogFile, header=0)
        else:
            log = pd.DataFrame(columns = ['epoch', 'dataName', 'pred_mask_size', 'valid_mask_size', 'pred_valid_maskSizeRatio', 
                                          'avgMaskRatePerInput', 'invalidNum', 'totNum', 'invalidRate'])
        log.loc[len(log)] = [epoch, dataName, predMaskSize, validMaskSize, maskSizeRatio,avgMaskRatePerInput, len(invalid), len(df), invalidRate]
        log.to_csv(maskLogFile, index=False)
        
        df['maskSize'] = df['valid_maskSize']
        df.drop(columns = ['valid_maskSize'], inplace=True)  

        return df
    
    def multi_mask_assigned(self, df, inFeat, outFeat):
        if self.debug: print("** multi_mask_assigned ==========")  
        df['masked_sent'] = df.apply(lambda x: self.replaceToken(x[inFeat], x[outFeat]), axis=1) 
        return df
    
    # 1. Retrieve pred_mask
    # 2. Check the validity of pred_mask
    # 3. Mask with pred_mask
    # 4. Save masked data (seperately: masked_train, masked_val)
    def genMaskedData(self, df, dataName, epoch, maskMode, source, targetFeat): 
        print("\n*** Generate masked_{} with {} ({})".format(dataName, dataName, len(df)))
        
        # Get predicted masks from the previous results
        path0 = "{}/epoch{}/masker/results/{}/".format(self.mainPath, epoch, source)
        predf = pd.read_csv("{}log_{}f0.csv".format(path0, dataName), header=0)
        if self.debug:
            print("predf:", predf.head(2))  
            print("df: ", df.head(2)['idx'])
            print("** Copy predicted masks to asmask_data - predf: {}, df: {}".format(len(predf), len(df)))
        
        df[self.maskFeat] = predf.pred
        df[self.maskFeat] = df[self.maskFeat].apply(lambda x: self.removeDuplicate(x)) # remove duplicated masks                
        df[self.maskFeat] = df[self.maskFeat].apply(lambda x: self.removeDupSpace(x))
        df = self.handleNullTokens(df, self.maskFeat)
        df = self.getMaskSize(df, maskFeat=self.maskFeat)
        df = self.getValidMask(df, epoch, dataName, self.maskFeat)
        
        df[self.maskFeat] = df['valid_mask'] # maskFeat = 'masked word', valid_mask
        df = self.getMaskSize(df, maskFeat=self.maskFeat)        
        
        df[self.mInFeat] = df[self.mInFeat].str.replace("Mask: ", "").str.replace("Classify: ", "") # mInFeat = Utterance
        
        
        if maskMode >=2 : df = self.multi_mask_assigned(df, self.mInFeat, self.maskFeat)  
        else: df = self.mask(df, targetFeat, random=0) # randomMode=0 when using predicted masks
        
        # save the updated asmask_data with new masks    
        path = "{}/epoch{}/masker/msl1/".format(self.mainPath, epoch)
        setPath(path)
        
        df[self.mInFeat] = "Mask: " + df[self.mInFeat].str.replace("Mask: ","") # 11/11 prevent "Mask: Mask: "
        df[self.masker_cols].sample(frac = 1).to_csv(path+"{}.csv".format(dataName), index=False)  
        return df 


    def getChangeSpeakerPair(self, df, inFeat, outFeat): # inFeat: Utterance or masked_sent, outFeat: self.cInFeat
        df['change_speaker'] = 0
        if 'Speaker' in df.columns:
            df.loc[df.Speaker != df.Speaker.shift(1) , 'change_speaker'] = 1
        else:
            df.loc[df[self.groupID] != df[self.groupID].shift(1) , 'change_speaker'] = 1

        df.loc[df.change_speaker==0, outFeat] = '@ ' + df.shift(1)[inFeat].fillna("") + ' @ '+ df[inFeat]
        df.loc[df.change_speaker==1, outFeat] = '# ' + df.shift(1)[inFeat].fillna("") + ' @ '+ df[inFeat]
        df[outFeat] = df[outFeat].str.replace("#  @", "@")
        return df
    
    
    # Make classification data 
    def makeTextClassificationData(self, epoch, mode):
        
        print(" == Make Classification data for {}".format(mode))
        ### Update CONTEXT in S-mem
        # self.smem = self.addContext(self.smem)
        #df[self.cInFeat] = 'Classify: ' + df[self.cInFeat]
        self.smem['masked_sent'] = self.smem['masked_sent'].str.replace("Classify: ", "")
        self.smem[self.cInFeat] = "Classify: " + self.smem['masked_sent'].str.replace("Classify: ", "")
        
        setPath(self.mainPath+"epoch{}/masker/msl1/".format(epoch)) 
        self.smem.to_csv(self.mainPath+"epoch{}/masker/msl1/smem_all.csv".format(epoch), index=False)
                
        path = '{}epoch{}/msl1/'.format(self.mainPath, epoch) 
        setPath(path)        
        
        if mode=='r_train':        # masked_train = r_train / masked_val = r_val
            masked_r_train = self.smem.loc[self.r_train.idx.values].copy(deep=True)
            masked_r_train[self.cInFeat] = "Classify: " + masked_r_train['masked_sent'].str.replace("Classify: ", "") # 8/8 cInFeat->masked_sent
            self.masked_r_val = self.smem.loc[self.r_val.idx.values]
            
            if True:
                if self.debug: print("R_val data: with masking")
                self.masked_r_val[self.cInFeat] = "Classify: " + self.smem.loc[self.r_val.idx.values, 'masked_sent'].values # 8/8 cInFeat->masked_sent
            else:
                if self.debug: print("R_val data: without masking")
                self.masked_r_val[self.cInFeat] = "Classify: " + self.smem.loc[self.r_val.idx.values, self.mInFeat].values # 8/8 cInFeat->masked_sent
            
            if self.debug:    
                print("* C-train:\n{}".format(masked_r_train.head(2)))
                print("* C-val:\n{}".format(self.masked_r_val[self.cls_cols].head(2)))           
            
            masked_r_train = masked_r_train.sample(frac=1)
            masked_r_train[self.cls_cols].to_csv(path+'masked_r_train.csv', index=False)
            self.masked_r_val[self.cls_cols].to_csv(path+'masked_r_val.csv', index=False)
             
    
        elif mode == 'i_train': # i_val로 early stopping for i_masker 
            masked_i_train = self.smem.copy(deep=True)
            masked_i_val = self.i_val.copy(deep=True)
            masked_i_train[self.cInFeat] = "Classify: " + masked_i_train['masked_sent'].str.replace("Classify: ", "") # 8/8 cInFeat->masked_sent
            
            if True: # with masked val data
                print("I_val data: with masking")
                masked_i_val[self.cInFeat] = 'Classify: ' + masked_i_val['masked_sent']
            else:   # without masking val data
                print("I_val data: without masking")                            
                masked_i_val[self.cInFeat] = 'Classify: ' + masked_i_val[self.mInFeat].str.replace("Classify: ", "")
           
            masked_i_train = masked_i_train.sample(frac=1)
            masked_i_train[self.cls_cols].to_csv(path+'masked_i_train.csv', index=False)
            masked_i_val[self.cls_cols].to_csv(path+'masked_i_val.csv', index=False)
            if self.debug:
                print("* U-train:\n{}".format(masked_i_train.head(2)))
                print("* U-val:\n{}".format(masked_i_val.head(2)))
         
        else: # 'test'
            masked_i_train = self.smem.copy(deep=True)
            masked_i_val = self.i_val.copy(deep=True)     
            masked_i_train[self.cInFeat] = 'Classify: ' + masked_i_train['masked_sent'].str.replace("Classify: ", "")  
            masked_i_val[self.cInFeat] = 'Classify: ' + masked_i_val['masked_sent'].str.replace("Classify: ", "") 
         
            masked_i_trainVal = pd.concat([masked_i_train[self.cls_cols], masked_i_val[self.cls_cols]], axis=0)
            masked_i_trainVal = masked_i_trainVal.sample(frac = 1)
            masked_i_trainVal.to_csv(path+'masked_i_trainVal.csv', index=False)       
            
            if self.maskTest: # with masked test data
                print("Test data: with masking")
                I_test = self.masked_i_test.copy(deep=True)   
                I_test[self.cInFeat] = 'Classify: ' + I_test['masked_sent']
                
            else:   # without masking test data
                I_test = self.i_test.copy(deep=True)
                print("Test data: without masking")                            
                I_test[self.cInFeat] = 'Classify: ' + I_test[self.mInFeat] 
                
            I_test[self.cls_cols].to_csv(path+'i_test.csv', index=False)   
            
            if True: #self.debug:
                print("* U-trainVal:\n{}".format(masked_i_trainVal.head(2)))
                print("* U-test:\n{}".format(I_test.head(2)))
                       

    def replace_df(self, a, b):
        return a.replace(b, '')
   

    def clearMem(self, path):
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(path) 

           
    def trainClassifier(self, pretrain, targetTrain, targetVal, targetTest, em_epoch, testOnly=0, clearM=1):
        if testOnly: trainEpoch=0
        else: trainEpoch=self.trainEpoch

        pathKey = '{}mask/{}/epoch{}/'.format(self.pathKey, self.modelName, em_epoch)
        args = Args(pathKey, mntPath=self.mntPath, pretrain=pretrain, targetX='clsInFeat', targetY='labels', 
                       targetTrain=targetTrain, targetVal = targetVal, targetTest=targetTest,
                       lr=self.lr, batch=self.batch, test_batch=self.test_batch, epoch=trainEpoch, keyword=self.keyword, 
                       tMask=self.tMask, maskMode=self.maskMode, saveLimit=self.saveLimit, groupID=self.groupID)
        bestAcc, acc, f1, mcc, cls_path, resFilePath = lm.main(args)

        if clearM and em_epoch > 1: 
            path = '{}data/{}mask/{}/epoch{}/model/'.format(self.mntPath, self.pathKey, self.modelName, em_epoch-self.patience-2) 
            if os.path.exists(path): self.clearMem(path)
        return bestAcc, acc, f1, mcc, cls_path, resFilePath        
            
    
    def trainMasker(self, em_epoch, pretrain, targetTrain, targetVal, targetTest, targetY='masked_word', clearM=1):
        print(" ===== Train a masker (train: {}, test: {}, pretrained masker: {})".format(targetTrain,targetTest, pretrain)) 
        pathKey='{}mask/{}/epoch{}/masker/'.format(self.pathKey, self.modelName, em_epoch)
        args = Args(pathKey, self.mntPath, pretrain, self.mInFeat, targetY, targetTrain, targetVal, targetTest, 
                    epoch=self.trainEpoch, batch=self.batch, test_batch=self.test_batch, lr=self.lr, keyword=self.keyword, 
                    tMask=self.tMask, maskMode=self.maskMode, saveLimit=self.saveLimit, groupID=self.groupID)
        _, _, _, _, maskerPath, resFilePath = lm.main(args) 
        print("Trained masker path: {}".format(maskerPath))
        if maskerPath == None:
            print(" !!! Error: fail to train a Masker\n\n")
            return None
        
        if clearM and em_epoch > 1: 
            path = self.mainPath+"epoch{}/masker/model/".format(em_epoch-2-self.patience-1)
            if os.path.exists(path): self.clearMem(path)
        return maskerPath
            

    
    def useMasker(self, em_epoch, maskerPath, targetTrain, targetVal, targetTest, targetY='masked_word'):
        pathKey='{}mask/{}/epoch{}/masker/'.format(self.pathKey, self.modelName, em_epoch)
        args = Args(pathKey, self.mntPath, maskerPath, self.mInFeat, targetY, targetTrain, targetVal, targetTest, 
                    epoch=0, batch=self.batch, test_batch=self.test_batch, lr=self.lr, keyword=self.keyword, 
                    tMask=self.tMask, maskMode=self.maskMode, saveLimit=self.saveLimit, groupID=self.groupID)       
        _ = lm.main(args)
    
    
    # When using this function, do not initRandomData every epoch because the indexes are changed.    
    def analMaskVal(self, df, dataName, epoch, maskWordFeat):  
        path = '{}epoch{}/'.format(self.mainPath, epoch)
        classifiRes = pd.read_csv('{}results/masked_r_train/log_masked_{}f0.csv'.format(path, dataName), header=0)
        classifiRes['correct'] = 0
        classifiRes.loc[classifiRes.label==classifiRes.pred, 'correct'] = 1
        #classifiRes.columns += str(epoch)    only for maskMode == 1

        mdata = pd.read_csv('{}msl1/masked_{}.csv'.format(path, dataName), header=0, usecols = [maskWordFeat]) # 'multi_mask' or 'masking_word'
        mdata.columns = ['pred_mask']
        alldata = pd.concat([classifiRes, mdata], axis=1)      
        print(" Epoch {}: {} classRes: {:.1f}".format(epoch, dataName, len(alldata[alldata['correct'] ==1])/len(alldata)*100))
        return alldata
    
    
    def updateSuccessMask(self, dataName, epoch, maskFeat): 
        self.smem[self.maskFeat] = self.smem[self.maskFeat].fillna(self.tMask)
        
        # Categorize mask instances based on the previous & current prediction results 
        cur_validataion_res = self.analMaskVal(self.masked_r_val, dataName, epoch, maskFeat)

        # Add original index to prediction ressults
        cur_validataion_res['idx'] = self.masked_r_val.idx.values
        if self.debug: print("cur_validataion_res:\n",  cur_validataion_res.head(2))
        cur_validataion_res.to_csv(self.mainPath +"epoch{}/cur_validataion_res.csv".format(epoch), index=False)
        
        # Distinguish successful and failed masks
        self.success_mask_idx = cur_validataion_res.loc[cur_validataion_res.correct==1, 'idx'].values
        self.fail_mask_idx = cur_validataion_res.loc[cur_validataion_res.correct==0, 'idx'].values        
        print("cur_validataion_res {} = masked_r_val {}".format(len(cur_validataion_res), len(self.masked_r_val)) )
        print("success_mask_idx {}, fail_mask_idx {}".format(len(self.success_mask_idx), len(self.fail_mask_idx)))
        
        self.smem.loc[self.r_val.idx, 'selected'] = 1  # Mark selected ones as validation data           
        ## Update S-mem with classified results
        if len(self.success_mask_idx) > 0:

            self.updateSuccessMasksToSmem(cur_validataion_res, maskFeat, epoch) ### UPDATE S-MASK to S-MEM 

        else: print(" !!!! No correct masks !!!!") 

        # 2) For failed masks, assign new random masks
        # For failed masks, reset masked_word and masked_sent 
        self.smem.loc[self.fail_mask_idx, 'masked_word'] = self.tMask  
        
        smem_size = len(self.smem.loc[self.smem['correct']==1])
        if False: # only use successful masks
            self.smem.loc[self.smem['correct']==1, self.masker_cols].to_csv(self.mainPath+"epoch{}/masker/msl1/smem_correct.csv".format(epoch), index=False)    
            print("Save smem for training S-Masker: {} ({:.1f})".format(smem_size, smem_size/len(self.smem)))
        else: # use successful masks + random masks
            self.smem[self.masker_cols].to_csv(self.mainPath+"epoch{}/masker/msl1/smem_correct_all.csv".format(epoch), index=False)
            print("* Save all smem for training S-Masker - success case: {} ({:.1f})".format(smem_size, smem_size/len(self.smem)))
        
        self.smem[self.masker_cols].to_csv(self.mainPath+"epoch{}/masker/msl1/i_masker_smem.csv".format(epoch), index=False)
        self.smem.to_csv(self.mainPath+"epoch{}/masker/msl1/smem_all.csv".format(epoch), index=False)
        self.saveSmemLog(cur_validataion_res, epoch)
        
        self.i_val = self.saveMaskerData(self.i_val, self.maskFeat, "i_masker_val", epoch)
        self.i_test = self.saveMaskerData(self.i_test, self.maskFeat, "i_masker_test", epoch)        

        
    def updateSuccessMasksToSmem(self, cur_validataion_res, maskFeat, epoch): # current allSuccess=0
        i_masked_val = cur_validataion_res.loc[cur_validataion_res.correct==1]
        success_idx = i_masked_val.idx.values     
        i_masked_val_valid = i_masked_val[i_masked_val['pred_mask']!=self.tMask]
        
        self.smem.loc[cur_validataion_res.idx.values, 'pred_mask'] = cur_validataion_res['pred_mask'].values  
        
        self.smem.loc[success_idx, 'correct'] = 1
        
        self.smem = self.getMaskSize(self.smem, maskFeat=self.maskFeat)
        self.smem = self.getValidMask(self.smem, epoch, 'smem', self.maskFeat) # get "valid_mask"
        self.smem['cur_valid_correct'] = 0 
        self.smem.loc[(self.smem.correct==1)&(self.smem.pred_mask!=self.tMask), 'cur_valid_correct'] = 1      
        
        self.smem = self.handleNullTokens(self.smem, self.maskFeat)        
        

        if True: # replace with current valid correct masks
            self.smem.loc[self.smem.cur_valid_correct==1, 'masked_word'] = self.smem.loc[self.smem.cur_valid_correct==1, "valid_mask"] 
        else:
            curValidCorrectMaskSize=self.smem.loc[self.smem.cur_valid_correct==1, 'maskSize'].mean()
            print("Added valid masks: {} (avg. maskSize: {:.1f})".format(len(self.smem.loc[self.smem.cur_valid_correct==1]), 
                                                                         curValidCorrectMaskSize))
            self.smem.loc[self.smem.cur_valid_correct==1, 'masked_word'] += " " +self.smem.loc[self.smem.cur_valid_correct==1, "valid_mask"] 
        
        self.smem = self.getMaskSize(self.smem, maskFeat=self.maskFeat)         
        self.smem.loc[(self.smem.correct==1)&(self.smem.masked_word!=self.tMask), 'valid_correct'] = 1
        self.smem[maskFeat] = self.smem[maskFeat].apply(lambda x: self.removeDuplicate(x))

        print(" ** Update S-MEM: All correct i_masks {}/{} ({:.1f}%), Valid correct i_masks {}/{} ({:.1f}%)".format(
            len(i_masked_val), len(self.smem), len(i_masked_val)/len(self.masked_r_val)*100, 
            len(i_masked_val_valid), len(self.masked_r_val), len(i_masked_val_valid)/len(self.masked_r_val)*100))
   
            
    def saveSmemLog(self, cur_validataion_res, epoch):
        correctNum = len(self.smem[self.smem.correct==1])
        selectedNum = len(self.smem[self.smem.selected==1])
        valid_correctNum = len(self.smem[self.smem.valid_correct==1])
        cur_val_suc_rate = np.round(len(self.success_mask_idx)/len(cur_validataion_res), 4)
        cur_val_fail_rate = np.round(len(self.fail_mask_idx)/len(cur_validataion_res), 4)
        smem_suc_rate = np.round(correctNum/len(self.smem), 4)
        smem_fail_rate = np.round((selectedNum-correctNum)/len(self.smem), 4)
        val_smem_suc_rate = np.round(valid_correctNum/len(self.smem), 4)
        selectedRate = len(self.smem[self.smem.selected==1])/len(self.smem)
        
        print("\n ## Current val: Success mask rate: {:.1f}, Fail rate: {:.1f}".format(cur_val_suc_rate*100, cur_val_fail_rate*100))        
        print("\n ## S-mem (total {}): Success mask rate: {:.1f} (valid: {:.1f}), Fail rate: {:.1f}".format(len(self.smem),
                                        smem_suc_rate*100, val_smem_suc_rate*100, smem_fail_rate*100))       
        print(" ## S-mem selected rate: {:.1f}\n".format(selectedRate*100))
       
        smemLogFile = self.mainPath + 'smemLog.csv'
        if os.path.exists(smemLogFile): sldf = pd.read_csv(smemLogFile, header=0)
        else:
            sldf = pd.DataFrame(columns=['epoch','smem_size','selected_rate','valid_success_mask_rate','success_mask_rate','fail_mask_rate', 'cur_val_success_rate'])
        sldf.loc[len(sldf)] = [epoch, len(self.smem),  selectedRate, val_smem_suc_rate, smem_suc_rate, smem_fail_rate, cur_val_suc_rate]
        sldf.to_csv(smemLogFile, index=False)
        


    ## Base =================================================================    
    def preprocFinalData(self, margs, randomMask):
        
        if randomMask:            
            train = self.loadPreprocRandomMaskData(margs.inputFeat, dataName='train', randomMask=1,removePunct=margs.removePunct, randomShuffle=1)    
            val = self.loadPreprocRandomMaskData(margs.inputFeat, dataName='val', randomMask=1,removePunct=margs.removePunct, randomShuffle=1)            
        else:
            train = self.loadPreprocBaseData(margs.inputFeat, dataName='train', removePunct=margs.removePunct, randomShuffle=1)    
            val = self.loadPreprocBaseData(margs.inputFeat, dataName='val', removePunct=margs.removePunct, randomShuffle=1)   
        
        trainVal = pd.concat([train, val], axis=0)  # due to dataPercent (need to keep validation part) -- already randomly masked
        trainVal = trainVal.sample(frac=1)
        targetData = '{}data/{}msl1/trainVal.csv'.format(self.mntPath, self.pathKey)    
        trainVal.to_csv(targetData, index=False)
                
        test = self.loadPreprocBaseData(margs.inputFeat, dataName='test', removePunct=margs.removePunct, randomShuffle=1)
        

    # search batch size and learning rate 
    def base_search_hyperParam(self, hyper, margs, search=1, randomMask=0):   
        
        if search: 
            print("============ Hyperparameter search ============")
            _ = self.loadPreprocBaseData(margs.inputFeat, dataName='train', removePunct=margs.removePunct)    
            _ = self.loadPreprocBaseData(margs.inputFeat, dataName='val', removePunct=margs.removePunct)   

            metricList = []
            for i in range(margs.startEpoch, len(hyper)):  # different random seed --> same results
                print(" ************ b = {}, lr = {} ************* ".format(hyper[i][0], hyper[i][1]))
                acc, f1, mcc, classifierPath, resFilePath = self.trainModel(margs, targetTrain='train', targetVal='val', targetTest='val', curEpoch=i,
                                                               batch=hyper[i][0], lr= hyper[i][1], targetX=margs.inputFeat)
                metricList.append(mcc)
                ut.clearMem(classifierPath)

            maxMetricIdx = metricList.index(max(metricList))
            bestBatch, bestLR = hyper[maxMetricIdx][0], hyper[maxMetricIdx][1] # best batch, lr
        else:
            bestBatch, bestLR = margs.batch, margs.lr

        print(" ============ Test =============")
        print(" ** Best hyperparameters: batch = {}, lr = {}".format(bestBatch, bestLR))

        accList, f1List, mccList = [], [], []
        for i in range(margs.randSeed, margs.endSeed):
            print("\n - Seed: {}".format(i))
            set_seed(i)
            train = self.loadPreprocBaseData(margs.inputFeat, dataName='train', removePunct=margs.removePunct, randomShuffle=1)    
            val = self.loadPreprocBaseData(margs.inputFeat, dataName='val', removePunct=margs.removePunct, randomShuffle=0)   
            test = self.loadPreprocBaseData(margs.inputFeat, dataName='test', removePunct=margs.removePunct, randomShuffle=0)
            
            acc, f1, mcc, classifierPath, resFilePath  = self.trainModel(margs, targetTrain='train', targetVal='val', targetTest='test', 
                                                                        curEpoch=i, batch=bestBatch, lr=bestLR, targetX=self.cInFeat)

            accList.append(acc)
            f1List.append(f1)
            mccList.append(mcc)
            #clearMem(classifierPath)
        print("\n ** Final: Average metrics: ")
        print(" Acc: {:.3f} ({:.3f})".format(np.mean(accList), np.std(accList)))
        print(" F1: {:.3f} ({:.3f})".format(np.mean(f1List), np.std(f1List)))
        print(" MCC: {:.3f} ({:.3f})".format(np.mean(mccList), np.std(mccList))) 
        return resFilePath
        
        
    ## R-Base =================================================================
    def getClassifierPath_rbase(self, margs, epoch):
        cpath = self.mntPath+"data/{}model/train_e{}/".format(margs.pathKey, epoch)
        checkpoint = os.listdir(cpath)
        checkpoint = [c for c in checkpoint if 'check' in c]
        classifierPath = cpath+"{}".format(checkpoint[0])
        return classifierPath

    # search optimal mask rate and iteration number, given batch size and learning rate 
    def randomMask_search_hyperParam(self, margs, maskRate, search=1, randomMask=0):   
        totStartTime = time.time()
        
        if search:
            print("============ Hyperparameter search: random mask rate ============")    
            val = self.loadPreprocRandomMaskData(margs.inputFeat, dataName='val', randomMask=0, removePunct=margs.removePunct)        
            adf = pd.DataFrame(columns = ['maskRateIdx','epoch', 'acc', 'f1', 'mcc'])
            for i in range(margs.startEpoch, len(maskRate)):
                print(" ----------------------------")
                print("\n\n ** maskRate = {}".format(maskRate[i]))        
                self.maskPercent = maskRate[i]

                set_seed(i + margs.randSeed*100)        
                train = self.loadPreprocRandomMaskData(margs.inputFeat, dataName='train', randomMask=1, removePunct=margs.removePunct)        

                cls_path_list =deque(maxlen=margs.patience+1)
                
                classifierPath = None
                for j in range(margs.endEpoch):

                    acc, f1, mcc, classifierPath, resFilePath  = self.trainModel(margs, targetTrain='train', targetVal='val', targetTest='val', curEpoch=j, 
                                                batch=margs.batch, lr=margs.lr, targetX='clsInFeat', pretrain=classifierPath)                
                    adf.loc[len(adf)] = [i, j, acc, f1, mcc] # i: maskRate index
                    
                    cls_path_list.append(classifierPath)

                    if j >= margs.warmup: # warmup = 3 epochs
                        earlyStop, esEpoch = ut.checkEarlyStop(adf, margs.warmup, margs.patience, 'mcc' )
                        if earlyStop: break

                ut.clearMem(classifierPath) 

            maxMetricIdx = adf.loc[adf.mcc.idxmax()].maskRateIdx
            bestMaskRate = maskRate[maxMetricIdx] 
            print("bestMaskRate: {}\nMetric list:\n".format(bestMaskRate, adf))        
        else:
            maxMetricIdx = "Given parameters"
            bestMaskRate = margs.maskPercent

        print(" ============ Test =============")
        print(" ** Best hyperparameters: maskRate = {}, trainEpoch: {} (maxMetricIdx: {})".format(bestMaskRate, margs.trainEpoch, maxMetricIdx))
        self.maskPercent = bestMaskRate

        # ------------------------------
        # Test ---- No random mask
        test = self.loadPreprocRandomMaskData(margs.inputFeat, dataName='test', randomMask= margs.maskTest, removePunct=margs.removePunct)        
        # ------------------------------

        cols = ['seed','bestMaskRate','batch', 'lr', 'epoch', 'f1', 'mcc', 'acc']
        allRes = pd.DataFrame(columns = cols)
        adfPath = "{}data/{}/model/adf.csv".format(self.mntPath, margs.pathKey)

        for seed in range(margs.randSeed, margs.endSeed):
            classifierPath = None
            cls_path_list = []

            if (margs.randSeed==seed) and (margs.startEpoch > 0):
                if os.path.exists(adfPath): adf = pd.read_csv(adfPath)
                classifierPath = self.getClassifierPath_rbase(margs, margs.startEpoch-1)
                print("* Restart with classifier: seed {} - {}".format(margs.randSeed, classifierPath))
                if margs.startEpoch==1: cls_path_list = [classifierPath]
                else: cls_path_list = [self.getClassifierPath_rbase(margs, i) for i in range(0, margs.startEpoch)] 

            else: adf = pd.DataFrame(columns = ['epoch', 'f1', 'mcc', 'acc'])

            if margs.startTest == 0: #seed > 1: 
                for i in range(margs.startEpoch, margs.endEpoch): # bestIterate
                    print(" ------------------------------------ ")
                    print("\n *** Seed {} - Iteration {}".format(seed, i))
                    set_seed(i + seed*100)   

                    train = self.loadPreprocRandomMaskData(margs.inputFeat, dataName='train', randomMask=1,removePunct=margs.removePunct, randomShuffle=1)    
                    val = self.loadPreprocRandomMaskData(margs.inputFeat, dataName='val', randomMask=1,removePunct=margs.removePunct, randomShuffle=0)   

                    acc, f1, mcc, classifierPath, resFilePath = self.trainModel(margs, targetTrain='train', targetVal ='val', targetTest='val', 
                                                    curEpoch=i, batch=margs.batch, lr=margs.lr, targetX='clsInFeat', pretrain=classifierPath)

                    cls_path_list.append(classifierPath)            
                    adf.loc[len(adf)] = [i, f1, mcc, acc]
                    adf.to_csv(adfPath, index=False)
                    if i >= margs.warmup:
                        earlyStop, esEpoch = ut.checkEarlyStop(adf, margs.warmup, margs.patience, 'mcc' )
                        if earlyStop: 
                            classifierPath = cls_path_list[esEpoch]
                            break
                        else: esEpoch = i

            if margs.startTest: 
                print("* Get checkpoint of the early stop epoch")
                esEpoch=margs.startTest   # early stop epoch     
                train = self.loadPreprocRandomMaskData(margs.inputFeat, dataName='train', randomMask=1,removePunct=margs.removePunct, randomShuffle=1)    
                val = self.loadPreprocRandomMaskData(margs.inputFeat, dataName='val', randomMask=1,removePunct=margs.removePunct, randomShuffle=0)   
                classifierPath = self.getClassifierPath_rbase(margs, margs.startTest)
                margs.startTest = False

            print("\n* Final Result ====== ")
            trainEpoch = margs.trainEpoch
            margs.trainEpoch=0
            targetData = '{}data/{}msl1/train.csv'.format(self.mntPath, self.pathKey)  
            print("\n!!! Train: {}, Test: {} - {}\n".format(len(train), len(test), targetData))
            final_acc, final_f1, final_mcc, classifierPath, resFilePath  = self.trainModel(margs, targetTrain='train', targetVal='val', targetTest='test', 
                                                    curEpoch=esEpoch, batch=margs.batch, lr=margs.lr, targetX='clsInFeat', pretrain=classifierPath)

            allRes.loc[len(allRes)] = [seed, bestMaskRate, margs.batch, margs.lr, esEpoch, final_f1, final_mcc, final_acc]
            allRes.to_csv("{}data/{}/res_{}.csv".format(self.mntPath, margs.pathKey, margs.keyword), index=False)
            print("\n ** Seed {}".format(seed))     
            print("Final results: F1: {:.3f}, MCC: {:.3f},  Acc: {:.3}\n".format(final_f1, final_mcc,final_acc))         
            #clearMem(classifierPath)
            margs.trainEpoch=trainEpoch
            margs.startEpoch= 0
            ut.printExpTime(totStartTime)

        path = '{}data/{}rbase/{}'.format(self.mntPath, margs.pathKey, self.modelName)

        cmodel_path = '{}/model/'.format(path)
        print(cmodel_path)
        if os.path.exists(cmodel_path): 
            print("Clear mem: ", cmodel_path)
            clearMem(cmodel_path)      
        return resFilePath

        
    def init_masker_smem(self, margs):
        if margs.startEpoch > 0:
            ppath = '{}data/{}/mask/{}/epoch{}/'.format(self.mntPath, margs.pathKey, margs.ppath, margs.startEpoch-1)

            checkpoint = os.listdir(ppath+'model/masked_i_train/')
            checkpoint = [c for c in checkpoint if 'check' in c]
            i_cls_path = ppath + 'model/masked_i_train/{}/'.format(checkpoint[0])

            if margs.startEpoch > 1:        
                checkpoint = os.listdir(ppath+'masker/model/smem_correct_all/')
                checkpoint = [c for c in checkpoint if 'check' in c]
                i_maskerPath = ppath + 'masker/model/smem_correct_all/{}/'.format(checkpoint[0])

                checkpoint = os.listdir(ppath+'masker/model/masker_train/')
                checkpoint = [c for c in checkpoint if 'check' in c]
                r_maskerPath = ppath + 'masker/model/masker_train/{}/'.format(checkpoint[0])

                checkpoint = os.listdir(ppath+'model/masked_r_train/')
                checkpoint = [c for c in checkpoint if 'check' in c]
                r_cls_path = ppath + 'model/masked_r_train/{}/'.format(checkpoint[0])

                i_masker_path_list = ['']*(margs.startEpoch-1) + [i_maskerPath] # no masker for epoch 0 
                i_cls_path_list = ['']*(margs.startEpoch-1)+ [i_cls_path]
                self.smem = pd.read_csv(ppath + 'masker/msl1/smem_all.csv', header=0)
            else:
                i_masker_path_list = [''] # no masker for epoch 0 
                i_cls_path_list = [i_cls_path]
                i_maskerPath = None
                r_maskerPath = None
                r_cls_path = None

            if os.path.exists(ppath+"adf.csv"):
                adf = pd.read_csv(ppath+"adf.csv")
            else: adf = pd.DataFrame(columns = ['epoch', 'acc', 'f1', 'mcc'])

        # if margs.maskerPath != '':
        #     r_maskerPath = '{}data/mask/{}'.format(mntPath, margs.maskerPath)
        #     i_maskerPath = '{}data/mask/{}'.format(mntPath, margs.maskerPath)
        else:
            if margs.startEpoch==1:
                ppath = '{}data/{}/mask/{}/epoch{}/'.format(self.mntPath, margs.pathKey, margs.ppath, margs.startEpoch-1)
                i_cls_path = ppath + 'model/masked_i_train/{}/'.format(checkpoint[0])
                i_cls_path_list = [i_cls_path]
            
            else: 
                i_cls_path = None
                i_cls_path_list = [] 
                                
            r_maskerPath = None
            i_maskerPath = None
            r_cls_path = None
            i_masker_path_list = [''] # no masker for epoch 0 
            adf = pd.DataFrame(columns = ['epoch', 'acc', 'f1', 'mcc'])

        self.i_train = self.loadPreprocData_DP(inputFeat=margs.inputFeat, dataName='train', removePunct=margs.removePunct)
        self.i_val = self.loadPreprocData_DP(inputFeat=margs.inputFeat, dataName='val', removePunct=margs.removePunct)    
        self.i_test = self.loadPreprocData_DP(inputFeat=margs.inputFeat, dataName='test', removePunct=margs.removePunct)
        
        return r_maskerPath, i_maskerPath, r_cls_path, i_cls_path, i_masker_path_list, i_cls_path_list, adf        
