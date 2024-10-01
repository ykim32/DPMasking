# Dual Process Masking, using Llama-2-7B/-3-8B
 
(+ baseline methods: no-masking, random masking)

Date: 06/13/2024

-------------------------------------------------------------
1. Development environment
    * OS: CentOS Linux 7 (Core)
    * miniconda install: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
    * Download a llama model 
       - https://llama.meta.com/llama-downloads/ 

    * packages         
        
        python=3.10.14
        numpy=1.26.4 
        pytorch-mutex=1.0

        transformers=4.38.2
        sentence-transformers=2.6.1
	datasets=2.18.0

	peft=0.10.0
	bitsandbytes=0.42.0

     * GPU: with QoRA
        - Llama-2-7B: works with RTX 4060, or better ones
        - Llama-3-8B: recommend to use A10, A30, L40 or better ones

----------------------------------------------------------------------
1. DP-Masking with Llama only for training classifier using pre-masked data

* Llama model fine-tuning, using duel-process masked input from DP-FT5

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

    -maskRate: the fraction of tokens masked in data (default=0.15)
        - This code dose not produce any masked data but use the data previously masked by DP-FT5 masker.
        - This parameter is only for differing the results.

* Other parameters for experiment setting
    -TRAIN/VAL/TEST: enable training, validation, test seperately 
    -fold : to load the masked input data from a specific fold (since we have cross valiation data sets) 
    -seed : set a random seed
    -keyword : use keyword to make the learned model and results distinguished
    -epochList : input 
    -bestEpoch : use the masked input data from the best epoch of DP-Masking 
    -evalType : default=epoch (option: steps)

--------------------------------------------------------------------------
2. Base Llama model

* Execution example
    $ python base_llama_finetune.py

* parameters for experiment setting
-model: model name
-TRAIN/VAL/TEST: enable training, validation, test seperately 
-fold : to load the masked input data from a specific fold (since we have cross valiation data sets) 
-seed : set a random seed
-keyword : use keyword to make the learned model and results distinguished
-valType: fixed with "unmasked" for base models without masking
-evalType : default=epoch (option: steps)
-patience : patience for early stopping


        
