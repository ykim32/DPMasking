import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, EarlyStoppingCallback, BitsAndBytesConfig
import transformers
from trl import SFTTrainer
from transformers import AutoTokenizer
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model # get_peft_config,
from sklearn.metrics import f1_score
import datasets

datasets.disable_progress_bar()


mntPath = "./llama/model/"                  #### Change the model root path
llama_model_name = "Llama-2-7b-chat-cps"    #### Change the new model name
#llama_model_name = "Llama-3-cps"

labels = ['Regulating', 'Negotiating', 'Maintaining', 'Sharing', 'Other', 'Off-task']
num_labels = ['1. Regulating', '2. Negotiating', '3. Maintaining', '4. Sharing', '5. Other', '6. Off-task']

def setPaths(modelName, args):    
    resPath = "data/{}/{}/".format(modelName, args.keyword)
    outModelPath = "model/{}/".format(modelName)
    modelID = "{}-{}-b{}-f{}-s{}-E{}-m{}".format(modelName, args.keyword, args.batch, args.fold, args.seed, args.epochList, int(args.maskPercent*100))
    savePath = "{}res_{}.csv".format(resPath, modelID)
    if not os.path.exists(outModelPath): os.makedirs(outModelPath)
    if not os.path.exists(resPath): os.makedirs(resPath)
    print("modelID: {}".format(modelID))

    # input files
    if 'base' in modelName:
        trainFile = "data/base/cv/{}/train.csv".format(args.fold)
        valFile = "data/base/cv/{}/val.csv".format(args.fold)
        testFile = "data/base/cv/{}/test.csv".format(args.fold)
    elif 'dp' in modelName:
        trainFile = "data/dp/f{}_s{}/epoch{}/masked_i_train.csv".format(args.fold, args.seed, args.bestEpoch)
        valFile = "data/dp/f{}_s{}/epoch{}/masked_i_val.csv".format(args.fold, args.seed, args.bestEpoch)
        testFile = "data/dp/f{}_s{}/i_test.csv".format(args.fold, args.seed)
    elif 'cum' in modelName:
        trainFile = "data/dp/f{}_s{}/masked_i_train_E{}.csv".format(args.fold, args.seed, args.epochList)
        if args.valType=='masked':
            valFile =  "data/dp/f{}_s{}/epoch{}/masked_i_val.csv".format(args.fold, args.seed, args.bestEpoch)
        else:
            valFile = "data/dp/f{}_s{}/val.csv".format(args.fold, args.seed)
        testFile = "data/dp/f{}_s{}/i_test.csv".format(args.fold, args.seed)
    elif 'rm' in modelName:
        trainFile = "data/rmask/rmask{}/{}/train_randMask{}.csv".format(int(args.maskPercent*100), args.fold, args.seed)
        valFile = "data/rmask/rmask{}/{}/val_randMask{}.csv".format(int(args.maskPercent*100), args.fold, args.seed)
        testFile = "data/rmask/rmask{}/{}/test.csv".format(int(args.maskPercent*100), args.fold)
    
    new_model = "{}{}-{}".format(outModelPath, llama_model_name, modelID)    

    print("train: {}".format(trainFile))
    print("val: {}".format(valFile))
    print("test: {}".format(testFile))
    print("new model: {}".format(new_model))
    return trainFile, valFile, testFile, savePath, outModelPath, new_model, resPath, modelID

def getPriorContext(text):
    return " : ".join(text.split(":")[:-1]) + " : "

def addMessage(FileName):
    df = pd.read_csv(FileName)
    df['Message'] = df['clsInFeat'].str.split(":")
    df['Message'] = df['Message'].str[-1]
    df['priorContext'] = df['clsInFeat'].str.replace("Classify: ", "").apply(lambda x: getPriorContext(x))  
    df.to_csv(FileName, index=False)
    print("Null message: {}".format(df.Message.isnull().any()))



def getPrompt(contextFeat, targetFeat, dataType, dataFile):
    system_prompt = "You are a dialog act classifier. Answer with a single label among Regulating, Negotiating, Maintaining, Sharing, Other, and Off-task."
    
    dataset = load_dataset("csv", data_files={"{}".format(dataType): "{}".format(dataFile)})     
    data = {"text": []}
    print("Make {} input".format(dataType))
    if dataType == 'test':
        for i in range(len(dataset[dataType])):
            data["text"].append(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                                {system_prompt}<|eot_id|> 
                                
                                <|start_header_id|>user<|end_header_id|>
                                {dataset[dataType][contextFeat][i].strip()} 
                                ### Message: {dataset[dataType][targetFeat][i].strip()} <|eot_id|> 
                                
                                <|start_header_id|>assistant<|end_header_id|>
                                ### Label: """) 
    else:
        for i in range(len(dataset[dataType])):
            cur_label = dataset[dataType]['labels'][i]
            data["text"].append(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                                {system_prompt}<|eot_id|> 
                                
                                <|start_header_id|>user<|end_header_id|>
                                {dataset[dataType][contextFeat][i].strip()} 
                                ### Message: {dataset[dataType][targetFeat][i].strip()} <|eot_id|> 
                                
                                <|start_header_id|>assistant<|end_header_id|>
                                ### Label: {labels.index(cur_label) + 1}. {cur_label}""")
        
    data = Dataset.from_dict(data)
    return data, dataset





def getLabel(text, label_tag):
    labels = ['Regulating', 'Negotiating', 'Maintaining', 'Sharing', 'Other', 'Off-task']
    label_ids = ['1', '2', '3', '4', '5', '6']
    idx = text.find(label_tag) + len(label_tag)
    text = text[idx:idx+100]
    ans = 'None'
    for l in labels:
        if l in text : 
            ans = l
            break
    if ans == 'None':
         for i in range(len(label_ids)):
            if label_ids[i] in text:
                ans = labels[i]
                break
    return ans, text


##========================================= 
## Fine-tuning 
##

bnb_config = BitsAndBytesConfig(
          load_in_8bit=True,
          bnb_8bit_compute_dtype=torch.bfloat16,
          bnb_8bit_quant_type="nf4",
          #load_in_4bit=True,
          #bnb_4bit_quant_type="nf4",
          #bnb_4bit_compute_dtype=torch.bfloat16,
)

def setTrainer(model_name_or_path, trainData, valData, batch, modelKey, deviceID=0, evalType='epoch', patience=1):
    peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
   
    if evalType=='epoch':
        training_args = TrainingArguments(
            output_dir = '{}data/model/{}/'.format(mntPath, modelKey),
            num_train_epochs = 10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            do_eval=True,
            load_best_model_at_end=True,
            disable_tqdm=True,
        )
    else:
        training_args = TrainingArguments(
            output_dir = '{}data/model/{}/'.format(mntPath, modelKey),
            max_steps=4000,
            eval_steps = 400,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps = 400,
            save_total_limit=1,
            do_eval=True,
            load_best_model_at_end=True,
            disable_tqdm=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        #device_map={"": deviceID},
    )
    #model.cuda(device=deviceID)    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=trainData,
        eval_dataset=valData,
        dataset_text_field="text",
        max_seq_length=512,
        dataset_batch_size = batch,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    return trainer


def loadModel(modelPath, deviceID=0):
    tokenizer = AutoTokenizer.from_pretrained(modelPath, token='token')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoPeftModelForCausalLM.from_pretrained(modelPath,
                low_cpu_mem_usage=True, 
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                #load_in_4bit=True, 
                #device_map={"": deviceID},
                token='token')
    #model.cuda(device=deviceID)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.generation_config.pad_token_ids = tokenizer.pad_token_id
    return tokenizer, model


def test(tokenizer, model, testData, dataset, dataType, savePath, label_tag):
    print("** {} size: {}".format(dataType, len(testData)))
    if os.path.exists(savePath):
        rdf = pd.read_csv(savePath)
        startIdx = len(rdf)
        print("read rdf - {}: {}".format(savePath, len(rdf)))
    else:
        rdf = pd.DataFrame(columns = ["label", "pred", "output"])
        startIdx = 0

    for i in range(startIdx, len(testData)):
        inputs = tokenizer(testData["text"][i], return_tensors="pt").to("cuda")
        outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"),
                  temperature = 0.1,
                  top_p=0, top_k=1,
                  attention_mask=inputs["attention_mask"],
                  max_new_tokens=5,
                  pad_token_id=tokenizer.eos_token_id)

        # Extract the answer part
        out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #print("out: {}".format(out))
        pred, out = getLabel(out, label_tag)
        rdf.loc[len(rdf)] = [dataset["test"]['labels'][i], pred, out]
        rdf.to_csv(savePath, index=False)

        acc = len(rdf[rdf.label==rdf.pred])/len(rdf)
        f1 = f1_score(rdf.label, rdf.pred,  average='macro')
        if i % 30==0:
            print("({}) [f1: {:.3f}, acc: {:.3f}] ** org label: {}\t pred: {}".format(i, f1, acc,dataset["test"]['labels'][i], pred))

    acc = len(rdf[rdf.label==rdf.pred])/len(rdf)
    f1 = f1_score(rdf.label, rdf.pred,  average='macro')
    print("** Final {} results: [f1: {:.3f}, acc: {:.3f}]".format(dataType, f1, acc))
    print("save pred result: {}".format(savePath))

    return f1, acc

def saveRes(args, f1, acc, dataType, resPath):
    if os.path.exists(resPath): adf = pd.read_csv(resPath)
    else: adf = pd.DataFrame(columns = ["fold", "seed","dataType", "valType", "bestEpoch",  "f1", "acc"])

    adf.loc[len(adf)] = [args.fold, args.seed, args.bestEpoch, dataType, args.valType, np.round(f1, 3), np.round(acc, 3)]
    adf.to_csv(resPath, index=False)  
