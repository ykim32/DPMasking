"""
* Dual-process masking 
  - main function
 1) FT5   : no-masking baseline    (maskMode: 0)
 2) R-FT5 : dynamic random masking (maskMode: 1)
 3) DP-FT5: dual-process masking   (maskMode: 2)

 Execution: See README.txt
 Date: 6/5/2024
"""

from transformers import set_seed
import warnings
warnings.simplefilter(action='ignore') 
import time
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
import logging
logging.disable(logging.CRITICAL) # disable CRITICAL, ERROR, WARNING, INFO and DEBUG logging everywhere
import shutil
import argparse
import pandas as pd
import numpy as np
import random
import train_lm as lm
import masker as mk
import util_mask as ut
from collections import deque


def getMainArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-splitMode', type=str, default='byRandom') # byRandom or byGroup
    parser.add_argument('-repeatR', type=int, default=1)    
    parser.add_argument('-r_rate', type=float, default=1)
    parser.add_argument('-startTest', type=int, default=0)
    parser.add_argument('-maskMode', type=int, default=2) # 0: random 1: single-masker 2: multi-masker
    parser.add_argument('-maskPercent', type=float, default=0.15)
    parser.add_argument('-saveLimit', type=int, default=1) # for RBase, saveLimit should be (1 + patience)
    parser.add_argument('-patience', type=int, default=3)
    parser.add_argument('-warmup', type=int, default=0)     
    parser.add_argument('-search', type=int, default=1) 
    parser.add_argument('-maskTest', type=int, default=0) 
    parser.add_argument('-dataPercent', type=float, default=1.0) 

    parser.add_argument('-groupID', type=str, default="classID")
    parser.add_argument('-speaker', type=str, default="")    
    parser.add_argument('-tMask', type=str, default="[M]")      
    parser.add_argument('-val_size', type=float, default=0.3) 
    parser.add_argument('-lr', type=float, default=0.0003)     
    parser.add_argument('-batch', type=int, default=8)
    parser.add_argument('-randSeed', type=int, default=0)
    parser.add_argument('-endSeed', type=int, default=1)
    parser.add_argument('-test_batch', type=int, default=32)
    parser.add_argument('-startEpoch', type=int, default=0)
    parser.add_argument('-endEpoch', type=int, default=20) 
    parser.add_argument('-trainEpoch', type=int, default=1) 
    
    parser.add_argument('-removePunct', type=int, default=0)    
    parser.add_argument('-truncRightAlign', type=int, default=1)    
    
    parser.add_argument('-mntPath', type=str, default='../data/')  #### !!!!--- Change the data path
    parser.add_argument('-maskerPath', type=str, default='')
    parser.add_argument('-ppath', type=str, default=None)
    parser.add_argument('-pathKey', type=str, default='')
    parser.add_argument('-resCopyPath', type=str, default='data/res/')
    parser.add_argument('-maxTokenNum', type=int, default=-1)
    parser.add_argument('-inputFeat', type=str, default='context')
    parser.add_argument('-keyword', type=str, default='')     
    parser.add_argument('-debug', type=int, default=0)
    
    parser.add_argument('-vmReset', type=int, default=20)
    args = parser.parse_args()  
    print(args)
    return args 
    
print("{}".format(datetime.datetime.now()))

margs = getMainArgs() 
set_seed(margs.randSeed)

mc = mk.Masker(margs)  
mc.saveHyper(margs)
totStartTime = time.time()

# ============================================================
# No Masking  
if margs.maskMode==0:
    print(" !!! Base: No masking !!!!")  
    # hyper = [batch, lr]
    hyper = [[4, 0.0001],[4, 0.0002],[4, 0.0003], [4, 0.0004], [4, 0.0005], 
             [8, 0.0001],[8, 0.0002],[8, 0.0003],[8, 0.0004], [8, 0.0005], 
             [16, 0.0001],[16, 0.0002],[16, 0.0003],[16, 0.0004], [16, 0.0005]] 
    resFilePath = mc.base_search_hyperParam(hyper, margs, search=margs.search, randomMask=0)
    
# ============================================================
# Dynamic Random Masking
# * batch size and learning rate should be given from baseline hyperparameter search
elif margs.maskMode==1: # repeat random masking 
    print(" !!! R-Base: Repeat random masking 10 times !!!!")
    maskRate = [0.05, 0.1, 0.15, 0.2]
    resFilePath = mc.randomMask_search_hyperParam(margs, maskRate, search=margs.search, randomMask=1)
    
# ============================================================
# Dual-Process Masker Learning
elif margs.maskMode >= 2:
    
    # 0. Init Masker and success memory
    r_maskerPath, i_maskerPath, r_cls_path, i_cls_path, i_masker_path_list, i_cls_path_list, adf = mc.init_masker_smem(margs)
    
    for em_epoch in range(mc.startEpoch, mc.endEpoch+1):
        print("\n ===============================================")
        print(" ******* Epoch {} [keyword: {}] ******* ".format(em_epoch, margs.keyword))
        set_seed(em_epoch + margs.randSeed*100) 
        
        # if you want to change the validataion data for masker every epoch, place initData here. 
        print(" Step 0) Init: random masking data")
        mc.initDataRandomMask(inputFeat=margs.inputFeat, splitMode=margs.splitMode, em_epoch=em_epoch) 
            
        print(" **** Train Epoch: {}  ".format(mc.trainEpoch))
        if em_epoch == 0: mc.initSmem()
        else: 
            if em_epoch==1: mc.initSmem() 
            
            mcc_list, r_maskerPath_inner = [], []
            for r in range(margs.repeatR): 
                # Update the masker based on the results from the previous epoch
                print("=================================================\n-- Repeat R # {} / {}".format(r, margs.repeatR))
                print("\n Step 1) Random mask")
                mc.initDataRandomMask_R(inputFeat=margs.inputFeat, splitMode=margs.splitMode, em_epoch=em_epoch)

                mc.preproc_R_Masker(em_epoch, sampleRate=margs.r_rate, sampleIdx=r) # add valid masks from s-mem to the new validataion set 

                print("\n Step 2) Train R-Masker, using masker_train, and Mask smem") # train a masker to predict next mask (better than random)
                if (em_epoch +1)% margs.vmReset==0: 
                    r_maskerPath = None # to avoid overfitting  
                    r_clas_path = None
                if em_epoch == 3 : #margs.warmup:
                    r_maskerPath = None
                    r_cls_path = None
                 
                r_maskerPath = mc.trainMasker(em_epoch,  pretrain = r_maskerPath, targetTrain = 'masker_train', targetVal='r_masker_smem',  
                                              targetTest = 'r_masker_smem', clearM=1) 
                r_maskerPath_inner.append(r_maskerPath)
                
                print(" Step 3) multi_maskData: Mask trainig/val utterances, using the learned masker")
                mc.smem = mc.genMaskedData(mc.smem, 'r_masker_smem', em_epoch, margs.maskMode, source='masker_train', targetFeat=mc.mInFeat)
                mc.makeTextClassificationData(em_epoch, mode='r_train') # using smem, generate/save context data by concatenating prior utter. & speaker

                print("\n ===============================================")
                print("\n Step 4) Train R-Classifier (input: masked_r_train, output: masked_r_val) for S-MEM update")
                _, _, _, mcc, r_cls_path, _ = mc.trainClassifier(pretrain=r_cls_path, targetTrain='masked_r_train', targetVal='masked_r_val', targetTest='masked_r_val',
                                                                       em_epoch=em_epoch, clearM=1)
                mcc_list.append(mcc)
                ut.printExpTime(totStartTime) 

                print("\n Step 5) Update S-MEM")
                # Update only successful instances to S-MEM based on the classification results 
                mc.updateSuccessMask(dataName='r_val', epoch=em_epoch, maskFeat='masked_word') # save i_masker_smem.csv 
                
                if margs.repeatR > 1 and r > 0 and mcc_list[-2] > mcc_list[-1]:
                    margs.repeatR -= 1
                    r_maskerPath = r_maskerPath_inner[mcc_list.index(max(mcc_list))]
                    break
                    
                    
            print("\n Step 6) Train I-Masker with S-mem")
            smem_source = 'smem_correct_all'
            i_maskerPath = mc.trainMasker(em_epoch, pretrain=i_maskerPath, targetTrain=smem_source, targetVal='i_masker_smem', targetTest = 'i_masker_smem', clearM=1)         
            i_masker_path_list.append(i_maskerPath)

            # Update the predicted masks to S-MEM + generate masked_sent
            mc.smem = mc.genMaskedData(mc.smem, 'i_masker_smem', em_epoch, margs.maskMode, source=smem_source, targetFeat=mc.mInFeat)

            # Mask the validataion data
            mc.useMasker(em_epoch, i_maskerPath, targetTrain=smem_source,targetVal='i_masker_val',  targetTest='i_masker_val')
            mc.masked_i_val = mc.genMaskedData(mc.i_val, 'i_masker_val', em_epoch, margs.maskMode, source=smem_source, targetFeat=mc.mInFeat)
            
        # Early stop (with validation accuracy)
        if (em_epoch >= 0):
            print("\n Step 7) Train U-Classifier (input: masked_i_train, output: masked_i_val) for U-validation")            
            mc.makeTextClassificationData(em_epoch, mode='i_train')
            
            bestAcc, acc, f1, mcc, i_cls_path, _ = mc.trainClassifier(pretrain=i_cls_path, targetTrain='masked_i_train', 
                                                                      targetVal = 'masked_i_val', targetTest='masked_i_val',
                                                                   em_epoch=em_epoch, clearM=1)
            i_cls_path_list.append(i_cls_path)            
            adf.loc[len(adf)] = [em_epoch, acc, f1, mcc]
            adf.to_csv(mc.mainPath+"epoch{}/adf.csv".format(em_epoch), index=False)
            
            # -----------------------------------------------------------
            print("\n !! Current U-Classifier Accurcy: {} / bestAcc: {:.1f}%".format(acc, bestAcc))
            if em_epoch >= margs.warmup:   
                earlyStop, esEpoch = ut.checkEarlyStop(adf, margs.warmup, margs.patience, 'mcc')
                if earlyStop: 
                    final_masker_path = i_masker_path_list[esEpoch]
                    final_cls_path = i_cls_path_list[esEpoch]                    
                else: # select the max 
                    final_cls_path = i_masker_path_list[adf.mcc.idxmax()]
                    final_masker_path = i_cls_path_list[adf.mcc.idxmax()]
                    
                print(" Final Classifier path: {}".format(final_cls_path))
                print(" Final Masker path: {}".format(final_masker_path))    
                
                if earlyStop or em_epoch == margs.endEpoch-1:  #if (em_epoch > 0): # and (acc < np.round(bestAcc/100,3)-0.001):
                    print("\n\n ******** Final results:")
                    
                    if margs.maskTest:
                        mc.useMasker(esEpoch, final_masker_path, targetTrain=smem_source, targetVal='i_masker_val', targetTest='i_masker_test')
                        mc.masked_i_test = mc.genMaskedData(mc.i_test, 'i_masker_test', esEpoch, margs.maskMode, source=smem_source,
                                                            targetFeat=mc.mInFeat)
                    
                    mc.makeTextClassificationData(esEpoch, mode='test')
                    
                    mc.trainEpoch=0 # test only
                    _, final_acc, final_f1, final_mcc, _, resFilePath = mc.trainClassifier(pretrain= final_cls_path, targetTrain='masked_i_train', 
                                                targetVal = 'masked_i_val', targetTest='i_test', em_epoch=esEpoch, clearM=1) #targetTrain='masked_i_trainVal', 
                    mc.trainEpoch=margs.trainEpoch
                    break

        ut.printExpTime(totStartTime)        

    # Clear all the saved models   
    if False: 
        path = margs.mntPath + 'data/' + margs.pathKey +'mask/' + mc.modelName
        for i in range(em_epoch+1):
            cmodel_path = '{}/epoch{}/model/'.format(path, i)
            mmodel_path = '{}/epoch{}/masker/model/'.format(path, i)
            if os.path.exists(cmodel_path): ut.clearMem(cmodel_path)    
            if os.path.exists(mmodel_path): ut.clearMem(mmodel_path)            

    print("\n * * Final results: Acc: {:.3}, F1: {}, MCC: {}".format(final_acc, final_f1, final_mcc)) 

ut.printExpTime(totStartTime) 

if not os.path.exists(margs.resCopyPath):
    os.makedirs(margs.resCopyPath)
shutil.copyfile(resFilePath, margs.resCopyPath + resFilePath.split('/')[-1])
