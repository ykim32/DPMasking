# Dual Process Masking (+ baseline methods: no-masking, dynamic random masking)

Date: 06/13/2024

-----------------------------------------------------------------------
1. Development evnrionment
    * OS: CentOS Linux 7 (Core)
    * miniconda install: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
    * packages (see the installation guide at the bottom):
        python=3.10.12
        numpy=1.26.4 
        pytorch=2.0.1
        transformers=4.38.2
        sentence-transformers=2.2.2
        datasets=2.10.1
        nvidia-ml-py3=7.352.0
        nltk=3.8.1

-----------------------------------------------------------------------
2. Execution
Data should be under the folder 'data/' (e.g.: data/oasis/t32/)

Example with Oasis corpus

1) No-masking baselilne:
   1.1) With hyperparameter search (-search=1)
    $ python main.py -inputFeat=context -search=1 -test_batch=32 -maskMode=0 -trainEpoch=20 -dataPercent=1 -pathKey=oasis/
    
   1.2) With fixed hyperparameters (-search=0, given batch and learning rate (-batch=X -lr=Y))
    $ python main.py -inputFeat=context -search=0 -test_batch=32 -batch=8 -lr=0.0004 -maskMode=0 -trainEpoch=20 -dataPercent=1 -pathKey=oasis/    
   
2) Dynamic random masking:

    $ python main.py -inputFeat=context -test_batch=32 -search=1 -maskMode=1 -trainEpoch=20 -maskPercent=0.05 -dataPercent=1 -pathKey=oasis/rbase/ -keyword=rbase_0
    
    2.2) With fixed hyperparameters (-search=0, given batch and learning rate (-batch=X -lr=Y))
    $ python main.py -inputFeat=context -test_batch=32 -search=0 -batch=8 -lr=0.0004 -maskMode=1 -trainEpoch=20 -dataPercent=1 -maskPercent=0.05 -pathKey=oasis/rbase/ -keyword=rbase_0


3) Dual-process masking:
   
    3.1) single cycle for training a reflective masker M_R
    $ python main.py -inputFeat=context -test_batch=32 -batch=8 -lr=0.0004 -maskMode=2 -trainEpoch=5 -maskPercent=0.15 -randSeed=0 -pathKey=oasis/ 

    3.2) multiple cycles for training a reflective masker M_R with n-percent of the training data 
     (e.g., -repeatR=2 -r_rate=0.5: repeat 2 cycles with 50% of the training data to learn a Masker before training the intuitive models) 
    $ python main.py -inputFeat=context -test_batch=32 -batch=8 -lr=0.0004 -maskMode=2 -trainEpoch=5 -maskPercent=0.15 -randSeed=0 -pathKey=oasis/ -repeatR=2 -r_rate=0.5 -keyword=R2r50


-----------------------------------------------------------------------
3. Main Hyperparameters:
    -maskMode        (0: no-masking baseline, 1: dynamic random masking,  2: dual-process maskiing)
    -maskPercent     (masking percent per input, default=0.15)
    -inputFeat       (input feature name, default='text')
    -groupID         (ID for seperating different dialogue groups, default=classID) 
    
    * for evaluation setting
    -pathKey         (data path excluding the top level 'data/' directory, default='')    
    -keyword         (any keyword to distinguish a model, default='')     
    -search          (hyperparameter search mode, 0: no search, 1: search, default=1) 
    -dataPercent     (percent of the data used, default=1.0 (range: 0 < dataPercent <=1)) 
    -patience        (for early stop of model training, default=3)    

    -val_size        (sub-validation size only for dual-process masking, default=0.3) 
    -lr              (learning rate, default=0.0003)     
    -batch           (batch size for training, default=8)
    -test_batch      (batch size for test data, default=16)
    
    -randSeed        (starting random seed, default=0)
    -endSeed         (ending random seed, default=1)
    -endEpoch        (end epoch of outer loop, default=20) 
    -trainEpoch      (max epoch of inner loop, default=10) # set more than 10 for the base approach and random masking, 5 or less for DP-masking
    
    -removePunct     (remove punctuation option, default=0)    
    -truncRightAlign (alignment of trucation, 0: left, 1: right, default=1)    
    
    -maxTokenNum     (-1: full token length, other integer: trunctae the max token length, default=-1) 
    
    -debug           (0: normal mode, 1: debue mode, default=0)


        
