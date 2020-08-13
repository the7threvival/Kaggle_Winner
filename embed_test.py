import os
import torch

import pandas as pd
import numpy as np
from models import *
from dataSet import *
from train import get_features

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# For ease of debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
num_TTA = 2

# To stop cv2/dataloader deadlock type situation
cv2.setNumThreads(0)

def transform(image, mask):
    raw_image = cv2.resize(image, (512, 256))
    raw_mask = cv2.resize(mask, (512, 256))
    raw_mask = raw_mask[:, :, None]
    raw_image = np.concatenate([raw_image, raw_mask], 2)
    images = []

    image = raw_image.copy()
    image = np.transpose(image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    images.append(image)

    image = raw_image.copy()
    image = np.fliplr(image)
    image = np.transpose(image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    images.append(image)

    return images

def embed_collate(batch):
    batch_size = len(batch)
    images = []
    labels = []
    names = []
    #print(labels)
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            #print(batch[b][1])
            images.extend(batch[b][0])
            labels.append(batch[b][1])
    images = torch.stack(images, 0)
    #print(labels)
    labels = torch.from_numpy(np.array(labels))
    return images, labels, names

def add_embeddings(CLASSES, model_name, fold_index, checkPoint_start, features_file):
    # Get the model
    device = torch.device('cuda')
    model = model_whale(num_classes=CLASSES * 2, inchannels=4, model_name=model_name).to(device)
    
    # Find result dir
    resultDir = './result/{}_{}'.format(model_name, fold_index)
    checkPoint = os.path.join(resultDir, 'checkpoint')
    
    # Load the pretrained weights
    if not checkPoint_start == 0:
        ckp = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (checkPoint_start)))
        model.load_state_dict(torch.load(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start))))
    
    # Load image data
    to_add = pd.read_csv('./input/embed_split_{}_add.csv'.format(fold_index))
    # Only do if necessary
    if 0:
        to_add = pd.read_csv('./input/embed_split_{}.csv'.format(fold_index))
        # Split up the embedding images and save to different files
        data_test = to_add[1::2]
        outfile = "./input/embed_split_{}_test.csv".format(fold_index)
        data_test.to_csv(outfile, index=None) 
        
        to_add = to_add[::2]
        outfile = "./input/embed_split_{}_add.csv".format(fold_index)
        to_add.to_csv(outfile, index=None)
    names_embed = to_add['Image'].tolist()
    labels_embed = to_add['Id'].tolist()
    batch_size = 16
    mode = 'embed'
    print("\nNumber of images to add:", len(names_embed))
        
    # Setup dataloader
    dst_embed = WhaleTestDataset(names_embed, labels_embed, mode=mode, transform=transform)
    dataloader_embed = DataLoader(dst_embed, shuffle=False, drop_last=False, batch_size=batch_size, num_workers=8, collate_fn=embed_collate)
   
    # Load the embeddings
    infile = "train_features{}.csv".format(features_file)
    embeddings = torch.Tensor(pd.read_csv(infile).to_numpy()).float()
    infile2 = "train_ids{}.csv".format(features_file)
    ids = torch.Tensor(pd.read_csv(infile2).to_numpy()).long()

    # Get the features to add
    new_ids, feats = get_features(dataloader_embed, model, CLASSES*2) 
    
    # Concatenate and save the features
    added_feats = torch.cat([embeddings, torch.Tensor(feats).float()], 0)
    added_ids = torch.cat([ids.view(-1), torch.Tensor(new_ids).long()], 0)
    outfile = "train_ids{}_added.csv".format(features_file)
    outfile2 = "train_features{}_added.csv".format(features_file)
    df1 = pd.DataFrame(added_ids.numpy())
    df2 = pd.DataFrame(added_feats.numpy())
    # Keep track of id, vector and some info about where this
    # was gotten (model, fold, iteration?)
    df1.to_csv(outfile, index=None)
    df2.to_csv(outfile2, index=None)
   
    print("Files {} and {} created with added ids and features.".format(outfile, outfile2)) 
    # Done 

if __name__ == '__main__':
    # Relevant vars
    CLASSES = 839
    model_name = "senet154"
    fold_index = "final2"
    checkPoint_start = 8600
    features_file = "_final2"
    
    # Sanity check. And for posterity
    print("Model name:", model_name)
    print("Fold:", fold_index)
    print("Checkpoint start:", checkPoint_start)
    print("Getting features (if embed) from files with ending:", features_file)
    
    # Do the thing
    add_embeddings(CLASSES, model_name, fold_index, checkPoint_start, features_file)
