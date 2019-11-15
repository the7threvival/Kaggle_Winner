# Kaggle Humpback Whale Identification Challenge 1st place

## Dependencies
- python==3.6
- torch==0.4.1
- torchvision==0.2.1
- opencv
- pandas
- matplotlib

conda install -c conda-forge python==3.6
conda install -c conda-forge opencv pandas matplotlib  
conda install -c pytorch pytorch torchvision cudatoolkit=10.1
or maybe (conda install -c pytorch pytorch==0.4.1 torchvision=0.2.1 cudatoolkit)

<br>

## Base solution  
https://www.kaggle.com/c/humpback-whale-identification/discussion/82366  

<br>

## Prepare Data  
## Competition dataset  
trainset    -> ./input/train  
testset    -> ./input/test  

## Mask
cd input  
unzip mask.zip  
download model_50A_slim_ensemble.csv(https://drive.google.com/file/d/1hfOu3_JR0vWJkNlRhKwhqJDaF3ID2vRs/view?usp=sharing)  into ./input  

This can be done from the command line with the following:   
<code> wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hfOu3_JR0vWJkNlRhKwhqJDaF3ID2vRs' -O model_50A_slim_ensemble.csv </code>

## Challenge data  
download data, then put them into input/  
https://www.kaggle.com/c/whale-categorization-playground/data  
https://www.kaggle.com/c/humpback-whale-identification/data   
(The second one should be more complete)   

To efficiently do this, install the official kaggle command line tool:   
``` 
pip install kaggle (or with conda install -c conda-forge kaggle)   
```
For this next step you should have a Kaggle account already created.  
Then **follow** the "API credentials" <a href="https://github.com/Kaggle/kaggle-api">guidelines</a> for kaggle to be able to automatically verify your identity.

You then need to register with this competition. You need to click the late submission button and follow the steps outlined. Then, you can use the kaggle module to download the datasets directly to your directory through the command line using the following lines:
```
kaggle competitions files humpback-whale-identification  
kaggle competitions download humpback-whale-identification -f train.zip   
kaggle competitions download humpback-whale-identification -f test.zip   
kaggle competitions download humpback-whale-identification -f train.csv   
```  
Then extract train.zip, test.zip, and masks.zip to the appropriate directory: train/, test/, and . .  

<br>

## Train  
line 301 in train.py  
step 1.   
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;            freeze = False  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;               model_name = 'senet154'  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;               min_num_class = 10  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             checkPoint_start = 0  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             lr = 3e-4  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             #until train map5 >= 0.98  

step 2.  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             freeze = True  
 &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;            model_name = 'senet154'  
  &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;           min_num_class = 0  
    &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;         checkPoint_start = best checkPoint of step 1  
     &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;        lr = 3e-4  

step 3.  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             freeze = True  
 &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;       model_name = 'senet154'  
   &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;     min_num_class = 0  
   &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;     checkPoint_start = best checkPoint of step 2  
   &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;     lr = 3e-5  

## Test  
line 99 in test.py  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;       checkPoint_start = best checkPoint of step 3  

