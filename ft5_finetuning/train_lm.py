"""
* Dual-process masking 
  - training a language model (default: Flan-T5-small)
 Date: 6/5/2024
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import time
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
import logging
logging.disable(logging.CRITICAL) # disable CRITICAL, ERROR, WARNING, INFO and DEBUG logging everywhere

import random
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset 
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, balanced_accuracy_score
import util_mask as ut

torch.cuda.empty_cache()
import gc


LANG_MODEL = "google/flan-t5-small"
#LANG_MODEL = "google/flan-t5-base"
#LANG_MODEL = "t5-small"

class TrainLM():
    debug = 0
    
    def __init__(self, args, device, tokenizer):
        self.device = device
        self.tokenizer = tokenizer
        self.speaker = 'teacher' # args.speaker
        self.groupID = args.groupID
        self.maskMode = args.maskMode
        self.tMask = args.tMask
        self.mntPath = args.mntPath
        self.pathKey = args.pathKey
        self.pretrain = args.pretrain
        self.targetTrain = args.targetTrain
        self.targetVal = args.targetVal
        self.targetTest = args.targetTest
        self.epoch = args.epoch
        self.curEpoch = args.curEpoch
        self.saveLimit = args.saveLimit
        self.targetX = args.targetX        
        self.targetY = args.targetY
        self.labelNames = []
        self.trainEpochs = args.epoch 
        self.keyword = args.keyword        
        self.modelName = self.pathKey.split('/')[-1] + '_{}'.format(self.keyword)

        self.pretrainCheckpoint = args.pretrain
        self.lr = args.lr       # for a fixed lr
        self.batch = args.batch # for a fixed batch
        self.maxlen = args.maxlen
        self.maxSeqLen = args.maxSeqLen

        self.modelID, self.saveLogFile, self.scorePath = '', '', ''
        self.outRootPath, self.modelPath, self.resPath = '','',''
        self.trainFile, self.valFile, self.testFile, self.allres_file, self.dataPath = '', '', '', '', ''
        self.setPaths()        
    
    def setModel(self, model, out_dir, train_data, eval_data):
        if model == None:
            model = AutoModelForSeq2SeqLM.from_pretrained(LANG_MODEL)
            model = model.to(self.device)

        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=model)

        training_args = Seq2SeqTrainingArguments(
            output_dir = out_dir,
            # evaluation_strategy = "steps",
            # eval_steps = 200,   # float: [0, 1) - ratio of total training steps
            # save_strategy = "steps",
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate = self.lr,
            per_device_train_batch_size = self.batch,
            per_device_eval_batch_size = self.batch,
            weight_decay = 0.01,
            save_total_limit = self.saveLimit,
            num_train_epochs = self.trainEpochs, 
            fp16 = False, # True,  # only available with CUDA
            warmup_steps=1000,
            #overwrite_output_dir=True,
            #seed=seed,
            save_steps=10000,
            load_best_model_at_end = True
        )

        early_stop = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)


        trainer = Seq2SeqTrainer(
            model = model,
            args = training_args,
            train_dataset = train_data, 
            eval_dataset = eval_data, 
            tokenizer = self.tokenizer,
            data_collator = data_collator,
            callbacks=[early_stop]
        )
        return model, trainer

    def preprocess_function(self, examples): # , tokenizer, maxlen, targetY 
        inputs = [doc for doc in examples[self.targetX]]
        model_inputs = self.tokenizer(inputs, max_length=self.maxlen, truncation=True, padding=True)

        # with tokenizer.as_target_tokenizer():  # current labels are class labels, but not sentence. 
        labels = self.tokenizer(examples[self.targetY], max_length=self.maxlen, truncation=True, padding=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def initDataModel(self, model, modelPath):
        # load dataset 
        ds = load_dataset("csv", data_files={"train": "{}".format(self.trainFile), "val": "{}".format(self.valFile)}) 
        print("Data size: train ({}), val ({})".format(len(ds['train']), len(ds['val'])))
        token_ds= ds.map(self.preprocess_function, batched=True) # tokenized 

        # model load if pretrainCheckpoint is not null
        if self.pretrainCheckpoint == None: model = None 
        else: model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrainCheckpoint,local_files_only=True).to(self.device) 

        if not os.path.exists(modelPath):
            os.makedirs(modelPath)
        model, trainer = self.setModel(model, out_dir=modelPath, train_data=token_ds["train"], eval_data = token_ds["val"])
        return ds, token_ds, model, trainer


    def removeDuplicate(self, a):
        return " ".join(set(str(a).split()))    
    
    def removeDupSpace(self, x):
        return " ".join(str(x).split())    
    
    def handleNullTokens(self, df, feat):
        df[feat] = df[feat].fillna(self.tMask)          
        df[feat] = df[feat].astype("str")
        df.loc[df[feat]=='nan', feat] = self.tMask
        df[feat] = df[feat].apply(lambda x: self.removeDuplicate(x))
        df[feat] = df[feat].str.replace(self.tMask+" ", "").str.replace(" "+self.tMask, "")
        df[feat] = df[feat].apply(lambda x: self.removeDupSpace(x))        
        return df
    
    
    def checkNull(self, path):
       # print(" == Check nulls for {} from data files.".format(self.targetY))
        df = pd.read_csv(path, header=0)
        if df[self.targetY].isnull().any():
            numNull = len(df[df[self.targetY].isnull()])
            df = self.handleNullTokens(df, self.targetY)
            print("\n !!! ... Find {} null values and fill with {}\n".format(numNull, self.tMask))
            df.to_csv(path, index=False)
        


    # Generate the outputs, given the test data    
    def test(self, model, testFile):
        ds = load_dataset("csv", data_files={"test":"{}".format(testFile)}) 
        rdf = pd.DataFrame(columns = [self.groupID, 'label', 'pred'])
        testIDs = pd.read_csv(testFile, usecols = [self.groupID], header=0).values.tolist()

        for idx in range(len(ds['test'])):
            test_utt = ds['test'][idx][self.targetX]
            input_ids = self.tokenizer(test_utt, return_tensors="pt").to(self.device).input_ids

            outputs = model.generate(input_ids, max_length = self.maxlen)
            pred_label = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            org_label = ds['test'][idx][self.targetY]

            rdf.loc[len(rdf)] = [testIDs[idx], org_label, pred_label]
            if idx % 20000 == 0:
                print(idx)
        if self.saveLogFile !='':
            rdf.to_csv(self.saveLogFile, index=False)
        return rdf

    
    def test_batch(self, model, testFile, test_batch_size=32):
        ds = load_dataset("csv", data_files={"test":"{}".format(testFile)}) 
        rdf = pd.DataFrame(columns = [self.groupID, 'label', 'pred'])

        self.checkNull(testFile)
        
        testIDs = pd.read_csv(testFile, usecols = [self.groupID], header=0).values.tolist()
        
        test_size = len(ds['test'])
        batch_num = int(np.ceil(test_size/test_batch_size))
        print("test: {}, batch size: {}, batch_num: {}".format(ds['test'].num_rows, test_batch_size, batch_num))
        
        for batchID in range(batch_num):
            idx = batchID*test_batch_size
            test_batch = ds['test'][idx:idx+test_batch_size][self.targetX]
            input_ids = self.tokenizer(test_batch, return_tensors="pt", padding=True).to(self.device).input_ids
            outputs = model.generate(input_ids, max_length = self.maxlen, do_sample=False)
            
            batch_res = pd.DataFrame(columns = [self.groupID, 'label', 'pred'])            
            batch_res[self.groupID] = testIDs[idx:idx+test_batch_size]
            batch_res['label'] = ds['test'][idx:idx+test_batch_size][self.targetY]
            batch_res['pred'] = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            rdf = pd.concat([rdf, batch_res], axis=0)

        rdf.reset_index(drop=True, inplace=True)
        rdf['pred'] = rdf.pred.fillna(self.tMask)  # not allowed null values for training
        
        if self.saveLogFile !='':
            rdf.to_csv(self.saveLogFile, index=False)
        return rdf     
    
    
    def setPaths(self):
        self.outRootPath = '{}data/{}'.format(self.mntPath, self.pathKey)
        self.modelPath = '{}model/'.format(self.outRootPath)  
        self.resPath = '{}results/'.format(self.outRootPath) 
        
        if 'epoch' in self.outRootPath:
            mainPath = '/'.join(self.outRootPath.split('/')[:-2]) # remove 'epochX/'
        else:
            mainPath = self.outRootPath[:-1]
        
        if self.targetY == 'labels': 
            self.allres_file = '{}/allres_{}.csv'.format(mainPath, self.modelName)
            print("\n* allres_file: {}".format(self.allres_file)) 
        else: self.allres_file = '{}/allres_masker.csv'.format('/'.join(self.outRootPath.split()[:-1]))
     
        
        #print(" * set path: root: {}, model: {}, result: {}".format(self.outRootPath,self.modelPath,self.resPath))
        if not os.path.exists(self.modelPath): os.makedirs(self.modelPath)
        if not os.path.exists(self.resPath): os.makedirs(self.resPath)

    def setDataPaths_noCV(self):
        print("\n*** noCV: {} ================================= ".format(self.pathKey))
        self.dataPath = '{}msl{}/'.format(self.outRootPath, self.maxSeqLen)
        self.trainFile = '{}{}.csv'.format(self.dataPath, self.targetTrain) # train or trainVal
        self.valFile = '{}{}.csv'.format(self.dataPath, self.targetVal) # train or trainVal
        self.testFile = '{}{}.csv'.format(self.dataPath, self.targetTest)    # val or test    
        if self.debug:
            print("trainFile: {}".format(self.trainFile))
            print("valFile: {}".format(self.valFile))
            print("testFile: {}".format(self.testFile))
            
    # set log file paths with current lr and batch   
    def setLogPath(self, fold):
        if self.maskMode == 1: # RBase
            self.modelID = "{}_e{}".format(self.targetTrain, self.curEpoch)
        else:
            self.modelID = "{}".format(self.targetTrain)
        self.saveLogFile = "{}/{}/log_{}f{}.csv".format(self.resPath, self.modelID, self.targetTest, fold)
        self.scorePath = '{}/{}/score_{}f{}.csv'.format(self.resPath, self.modelID, self.targetTest, fold)
        if not os.path.exists(self.resPath+self.modelID): os.makedirs(self.resPath+self.modelID) 

        
    def train(self, model):
        path = self.modelPath + self.modelID
        if self.debug: print("Model path: {}".format(path))
      
        # check null values
        self.checkNull(self.trainFile)
        self.checkNull(self.valFile)
        
        ds, token_ds, model, trainer = self.initDataModel(model, path)        
            
        if self.trainEpochs>0:
            result = trainer.train()
            checkpointEpoch =  [i for i in os.listdir(path) if 'checkpoint' in i]
            print("checkpointEpoch:", checkpointEpoch)
            checkpointPath = trainer.state.best_model_checkpoint
            print("modelPath: ", checkpointPath)        
        else: 
            result = None
            checkpointPath = self.pretrainCheckpoint
        del ds
        gc.collect()
        
        return model, result, checkpointPath        

# End of class --------------------------------------------------------------------------------------------------------        
        
def main(args):    
    #print("{}\npathKey: {}".format(datetime.datetime.now(), args.pathKey))
    #tokenizer = AutoTokenizer.from_pretrained('t5-small', use_fast=True) 
    tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # if use device and .map with batched mode, errors occurs

    
    tra = TrainLM(args, device, tokenizer)

    totStartTime = time.time()                   
    # Fixed training and test data set (no cross validataion) ---------------------------------
    # tra.lr, tra.batch = args.lr, args.batch 
    
    tra.setDataPaths_noCV()
    labelNames = pd.read_csv(tra.trainFile, header=0)[args.targetY].unique().tolist()
    if args.targetY == 'labels':
        tra.labelNames = sorted(labelNames)
        if tra.debug: print(tra.labelNames)

    tra.setLogPath(fold=0)

    
    # -------------------------------------
    # Training
    model, result, checkpointPath = tra.train(model=None) 
    
    # -------------------------------------
    # Test
    rdf = tra.test_batch(model, tra.testFile, test_batch_size=args.test_batch)
    
    # -------------------------------------
    # Evaluation
    if (args.targetY == 'labels') and (args.epoch > 0):
        bestAcc, acc, f1, mcc = ut.evaluate_noCV(rdf, tra.labelNames, tra.lr, tra.batch, args.targetTrain, args.targetTest, 
                             path=tra.resPath + tra.modelID +'/', allres_file=tra.allres_file, curEpoch=args.curEpoch)    
        print(" === Trained a Classifier")
        
    elif (args.targetY == 'labels') and (args.epoch == 0):
        bestAcc, acc, f1, mcc = ut.evaluate_noCV(rdf, tra.labelNames, tra.lr, tra.batch, args.targetTrain, args.targetTest, 
                             path=tra.resPath + tra.modelID +'/', allres_file=tra.allres_file, curEpoch=args.curEpoch)
        print(" === Tested a Classifier")
    
    else:
        bestAcc, acc, f1, mcc =0.0, 0.0, 0.0, 0.0        
        if (args.epoch > 0): print(" === Trained a Masker")
        else: print(" === Tested a Masker")


    totLearnTime = (time.time() - totStartTime)/60 
    print("Total Time: {:.1f} min ({:.1f} hours)".format(totLearnTime, totLearnTime/60))
    return bestAcc, acc, f1, mcc, checkpointPath, tra.allres_file
