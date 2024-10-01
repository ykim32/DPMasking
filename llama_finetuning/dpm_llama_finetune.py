"""
Duel Process Masking, using Llama models
 
* License: The use of Llama code is under Meta’s Community License Agreement.

* Llama model fine-tuning, using duel-masked input from DP-FT5

  1) The training code is the same to training base Llama models.
  2) The management code for the masked training data from previously learnd DP-Masker is added.

* Execution example
    $ python dpm_llama_finetune.py -model=dp  -valType=masked 
                                   -fold=0 -seed=0 -evalType=epoch -keyword=Eb_10 -epochList=b_10 -bestEpoch=10

* Main parameters for DP-Masking
    -model : training mode 
        1) base: base Llama model training (data folder: "data/base/cv/")
        2) rm: random masking (data folder: 'data/rmask/rmask')
        3) dp: dual-process masking using pre-masked input by DP-FT5, see dp_masking_code to use DP-FT5 (data folder: 'data/dp/f{fold id}_s{random seed}/')
        
    -valType : masking mode for validation 
        1) masked: use "masked input" for validation, while we use "unmasked input" for test (inference)
            - required pre-masked validation data
        2) no_mask: use "unmasked, original input" for validation like we use "unmasked input" for test (inference) 
            - required unmasked validation data

* Other parameters for experiment setting
-TRAIN/VAL/TEST: enable training, validation, test seperately 
-fold : to load the masked input data from a specific fold (since we have cross valiation data sets) 
-seed : set a random seed
-keyword : use keyword to make the learned model and results distinguished
-epochList : input 
-bestEpoch : use the masked input data from the best epoch of DP-Masking 
-evalType : default=epoch (option: steps)

"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"
import warnings
warnings.filterwarnings("ignore")
import argparse
from transformers import set_seed
import util as ut
import torch
import datasets

datasets.disable_progress_bar()


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fold', type=int, default=0) 
    parser.add_argument('-seed', type=int, default=0)    
    parser.add_argument('-batch', type=int, default=4)
    parser.add_argument('-TRAIN', type=int, default=1)
    parser.add_argument('-VAL', type=int, default=1)
    parser.add_argument('-TEST', type=int, default=1) 
    parser.add_argument('-keyword', type=str, default="")
    parser.add_argument('-bestEpoch', type=int, default=10) 
    parser.add_argument('-maxEpoch', type=int, default=3)
    parser.add_argument('-epochList', type=str)
    parser.add_argument('-model', type=str, default='dp')
    parser.add_argument('-valType', type=str, default='masked')
    parser.add_argument('-evalType', type=str, default='epoch')
    parser.add_argument('-patience', type=int, default=1)
    parser.add_argument('-deviceID', type=str, default="0")
    parser.add_argument('-maskPercent', type=float, default=0.15)
    args = parser.parse_args()  
    print(args)
    return args 

args = getArgs()

#torch.cuda.set_device(args.deviceID)
print("= GPU: {}".format(torch.cuda.current_device()))

modelName = args.model
contextFeat = "priorContext"
targetFeat = "Message"

loaded_model = ""
model_path = "./model/meta-llama/"   #### !!!!! ---- Change the Llama model root path
model = model_path + "Llama-2-7b-chat-hf"  #### !!!!! ---- Change the base Llama model name
#model = model_path + "Meta-Llama-3-8B-Instruct"   
print("base model: {}".format(model))

label_tag = "### Label: "
print("{} - Fold: {}, seed: {}, batch: {}, keyword: {}".format(modelName, args.fold, args.seed, args.batch, args.keyword))

trainFile, valFile, testFile, savePath, outModelPath, new_model, resPath, modelID = ut.setPaths(modelName, args)


def getDataEpoch(fold, seed, epoch, valType='masked'):    
    trainFile = "data/dp/f{}_s{}/epoch{}/masked_i_train.csv".format(fold, seed, epoch)
    if valType=='masked':
        valFile = "data/dp/f{}_s{}/epoch{}/masked_i_val.csv".format(fold, seed, epoch)
    else:
        valFile = "data/dp/f{}_s{}/val.csv".format(fold, seed)
    return trainFile, valFile

# training 
if args.TRAIN:
    set_seed(args.seed)
    
    trainData, _ = ut.getPrompt(contextFeat, targetFeat, dataType='train', dataFile = trainFile)
    valData, _ = ut.getPrompt(contextFeat, targetFeat, dataType='val', dataFile = valFile)
    if True: 
        trainer = ut.setTrainer(model, trainData, valData, args.batch, modelID, args.deviceID, args.evalType, args.patience)
        if loaded_model != "":
            _, loaded_model = ut.loadModel(model)
            trainer.model=loaded_model
    else: # load an exising model
        tokenizer, model = ut.loadModel(new_model)
        trainer = ut.updateTrainer(model, trainData, valData, args.batch, modelID, args.evalType)

    print(" **** Fine-tuning start: epoch{}".format(epoch))
    trainer.train()
    
    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)
    print("saved model: {}".format(new_model))

    del trainData, valData, trainer   
   
    
# ==============================
if args.VAL or args.TEST:
    tokenizer, model = ut.loadModel(new_model, args.deviceID)
    resPath="data/cum/{}/res_cum_all.csv".format(args.keyword)

if args.VAL:
    print(" **** Val: {} - {} ".format(modelID, new_model))  
    if args.valType=='masked': saveValPath = savePath[:-4]+"_val_masked.csv"
    else:  saveValPath = savePath[:-4]+"_val.csv"
    
    valData, dataset = ut.getPrompt(contextFeat, targetFeat, dataType='test', dataFile = valFile)
    f1, acc = ut.test(tokenizer, model, valData, dataset, dataType='test', savePath=saveValPath, label_tag=label_tag)
    ut.saveRes(args, f1, acc, "val", resPath)
        
if args.TEST:
    print(" **** Test: {} - {} ".format(modelID, new_model))  
    testData, dataset = ut.getPrompt(contextFeat, targetFeat, dataType='test', dataFile = testFile)
    f1, acc = ut.test(tokenizer, model, testData, dataset,  dataType='test', savePath=savePath, label_tag=label_tag)
    ut.saveRes(args, f1, acc, "test", resPath)

    
