"""
Llama model fine tuning using for baselines

* execution example
$ python base_llama_finetune.py
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import random
import argparse
from transformers import set_seed
from sklearn.metrics import f1_score

import util as ut

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fold', type=int, default=0) 
    parser.add_argument('-seed', type=int, default=0)    
    parser.add_argument('-batch', type=int, default=4)
    parser.add_argument('-TRAIN', type=int, default=1)
    parser.add_argument('-TEST', type=int, default=1) 
    parser.add_argument('-keyword', type=str, default="q8")
    parser.add_argument('-bestEpoch', type=int, default=10) 
    parser.add_argument('-maxEpoch', type=int, default=3)
    parser.add_argument('-epochList', type=str)
    parser.add_argument('-model', type=str, default='base')
    parser.add_argument('-valType', type=str, default='masked')
    parser.add_argument('-evalType', type=str, default='epoch')
    parser.add_argument('-patience', type=int, default=1)
    parser.add_argument('-maskPercent', type=float, default=0)
    args = parser.parse_args()  
    print(args)
    return args 

args = getArgs()
args.keyword = "{}-b{}".format(args.keyword, args.batch) 

modelName = args.model 
contextFeat = "priorContext"
targetFeat = "Message"

model_path = "./model/meta-llama/"  #### !!!!! ---- Change the Llama model root path 
model_name_or_path = "{}Llama-2-7b-chat-hf".format(model_path)   #### !!!!! ---- Change the base Llama model name
#model_name_or_path = "{}Meta-Llama-3-8B-Instruct".format(model_path) 

label_tag = "### Label:"

print("{} - Fold: {}, seed: {}, batch: {}, keyword: {}".format(modelName, args.fold, args.seed, args.batch, args.keyword))

trainFile, valFile, testFile, savePath, outModelPath, new_model, resPath, modelID = ut.setPaths(modelName, args)


# training 
if args.TRAIN:
    set_seed(args.seed)
    
    trainData, _ = ut.getPrompt(contextFeat, targetFeat, dataType='train', dataFile = trainFile)
    valData, _ = ut.getPrompt(contextFeat, targetFeat, dataType='val', dataFile = valFile)

    trainer = ut.setTrainer(model_name_or_path, trainData, valData, args.batch, modelID)
    
    del trainData, valData
    print(" **** Fine-tuning start")
    trainer.train()
    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)
    print("saved mode: {}".format(new_model))

    del trainer 



# ==============================
if args.TEST:
    # if args.TRAIN==0:
    #     #new_model = "model/base/Llama-2-7b-chat-hf-cps-base-ENone-b7b-q8-b4-f0-s1/"
    #     new_model = "model/base/Llama-2-7b-chat-hf-cps-base-ENone-b7b-q8-b4-f2-s0/"
    tokenizer, model = ut.loadModel(new_model)
    
    print(" **** Test: {} - {} ".format(modelID, new_model))
    
    testData, dataset = ut.getPrompt(contextFeat, targetFeat, dataType='test', dataFile = testFile)
    f1, acc = ut.test(tokenizer, model, testData, dataset,  dataType='test', savePath=savePath, label_tag=label_tag)

    allResPath = resPath+"res_base_all.csv"
    if os.path.exists(allResPath): adf = pd.read_csv(allResPath)
    else: adf = pd.DataFrame(columns = ["fold", "seed", "f1", "acc"])
    
    adf.loc[len(adf)] = [args.fold, args.seed, f1, acc]
    adf.to_csv(allResPath, index=False) 
    print(adf)    
        
