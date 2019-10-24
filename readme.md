# Kaggle Humpback Whale Identification Challenge  1st place code

#### Recent Updates
[2019.3.1 16:44] uploading mask file
#### Dependencies
- python==3.6
- torch==0.4.1
- torchvision==0.2.1


## Solution  
https://www.kaggle.com/c/humpback-whale-identification/discussion/82366  

### prepare data  
## competition dataset  
trainset    -> ./input/train  
testset    -> ./input/test  

## mask
cd input  
unzip mask.zip  
download model_50A_slim_ensemble.csv(https://drive.google.com/file/d/1hfOu3_JR0vWJkNlRhKwhqJDaF3ID2vRs/view?usp=sharing)  into ./input
This can be done from the command line with the following:
<code> wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hfOu3_JR0vWJkNlRhKwhqJDaF3ID2vRs' -O model_50A_slim_ensemble.csv </code>

## playground data  
download playground data, then put them into input/train  
https://www.kaggle.com/c/whale-categorization-playground/data  
To efficiently do this, you need to create a Kaggle account and be registered with this competition. You need to click the late submission button and follow the steps outlined. Then, you can use the kaggle module to download the datasets directly to your directory through the command line using the following lines:  
<code>   
pip install kaggle (or with conda)   
kaggle competitions files humpback-whale-identification  
kaggle competitions download humpback-whale-identification -f train.zip   
kaggle competitions download humpback-whale-identification -f test.zip   
kaggle competitions download humpback-whale-identification -f train.csv   
</code>

### Train  
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

### Test  
line 99 in test.py  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;       checkPoint_start = best checkPoint of step 3  

