from torch.utils.data import Dataset
import random
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import math
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time

BASE_SIZE = 256
# NUM_CLASSES = 5004 * 2
NUM_CLASSES = 839 * 2
#new_whale = True
new_whale = True

def do_length_decode(rle, H=192, W=384, fill_value=255):
    mask = np.zeros((H,W), np.uint8)
    if type(rle).__name__ == 'float': return mask
    mask = mask.reshape(-1)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0]-1
        end = start + r[1]
        mask[start : end] = fill_value
    mask = mask.reshape(W, H).T   # H, W need to swap as transposing.
    return mask

class WhaleDataset(Dataset):
    def __init__(self, names, labels=None, mode='train', transform_train=None,  min_num_classes=0, newWhale=True):
        super(WhaleDataset, self).__init__()
        global new_whale 
        new_whale = newWhale
        #print(new_whale, newWhale)
        self.pairs = 2
        self.names = names
        # labels is used to have all the labels that are input. they might
        # be repeated or not. (Represents original entries of elephant IDs
        # passed in as the training data)
        # By the end, repeated labels are removed
        self.labels = labels
        self.mode = mode
        self.transform_train = transform_train
        # this dict stores (label, id) pair for use when comparing results
        # from model prediction
        self.labels_dict = self.load_labels()
        self.bbox_dict = self.load_bbox()
        self.rle_masks = self.load_mask()
        # This is not used in this file and prob never anywhere else because
        # labels could still change...
        #self.id_labels = {Image:Id for Image, Id in zip(self.names, self.labels)}
        labels = []
        for label in self.labels:
            if label.find(' ') > -1:
                labels.append(label.split(' ')[0])
            else:
                labels.append(label)
        self.labels = labels
        if mode in ['train', 'valid']:
            # dict_train holds the list of images associated with a particular ID for
            # all IDs
            self.dict_train = self.balance_train()
            # This is overwritten right after so removed for now
            # self.labels = list(self.dict_train.keys())

            # Making sure there is a minimum number of classes for chosen IDs
            self.labels = [k for k in self.dict_train.keys()
                            if len(self.dict_train[k]) >= min_num_classes]
            images = [len(self.dict_train[k]) for k in self.dict_train.keys()
                            if len(self.dict_train[k]) >= min_num_classes]
            self.num_images = sum(images)

    def load_mask(self):
        print('loading mask...')
        rle_masks = pd.read_csv('./input/model_50A_slim_ensemble.csv')
        rle_masks = rle_masks[rle_masks['rle_mask'].isnull() == False]
        rle_masks.index = rle_masks['id']
        del rle_masks['id']
        rle_masks = rle_masks.to_dict('index')
        return rle_masks

    def load_bbox(self):
        # Image,x0,y0,x1,y1
        print('loading bbox...')
        bbox = pd.read_csv('./input/bboxs.csv')
        Images = bbox['Image'].tolist()
        x0s = bbox['x0'].tolist()
        y0s = bbox['y0'].tolist()
        x1s = bbox['x1'].tolist()
        y1s = bbox['y1'].tolist()
        bbox_dict = {}
        for Image,x0,y0,x1,y1 in zip(Images,x0s,y0s,x1s,y1s):
            bbox_dict[Image] = [x0, y0, x1, y1]
        return bbox_dict

    def load_labels(self):
        label = pd.read_csv('./input/label.csv')
        labelName = label['name'].tolist()
        dict_label = {}
        id = 0
        for name in labelName:
            if name == 'new_whale':
                dict_label[name] = NUM_CLASSES
                continue
            dict_label[name] = id
            id += 1
        return dict_label
    
    def balance_train(self):
        dict_train = {}
        for name, label in zip(self.names, self.labels):
            if not label in dict_train.keys():
                dict_train[label] = [name]
            else:
                dict_train[label].append(name)
        return dict_train
    
    def __len__(self):
        return len(self.labels)

    def get_image(self, name, transform, label, mode='train'):
        #import ipdb
        #ipdb.set_trace(context=5)
        
        #print()
        #print("In get_image")
        #print(name, label)
        image = cv2.imread('./input/{}/{}'.format(mode, name))
        # for Pseudo label
        if image is None:
            image = cv2.imread('./input/test/{}'.format(name))
        try:
            mask = do_length_decode(self.rle_masks[name.split('.')[0]]['rle_mask'])
            mask = cv2.resize(mask, image.shape[:2][::-1])
        except:
            mask = cv2.imread('./input/masks/' + name, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros_like(image[:,:,0])
        #print("got mask and image")
        
        #print("Trying to get bbox")
        # Get bounding box if it exists, else use whole image
        if name in self.bbox_dict: 
        #if False: 
            #print("got bounding box")
            x0, y0, x1, y1 = self.bbox_dict[name]
            image = image[int(y0):int(y1), int(x0):int(x1)]
            mask = mask[int(y0):int(y1), int(x0):int(x1)]
            #print(y0, y1, x0, x1)
            #print()
        
        #print(mask, image)
        #print("trying to run transform and getting stuck")

        #add_ is a number added to the label integer to create a new label
        #for flipped images, if flipped.. (smart)
        image, add_ = transform(image, mask, label)
        #print(label)
        #print("got image, returning")
        return image, add_

    def __getitem__(self, index):
        #import ipdb
        #ipdb.set_trace(context=5)
        label = self.labels[index]
        names = self.dict_train[label]
        nums = len(names)
        # IT SEEMS LIKE WHEN THIS LABEL IS 'NEW_WHALE', IT WILL GIVE BOTH A 
        # POSITIVE AND NEGATIVE OTHER NEW_WHALE AS WELL... THAT MAKES NO SENSE
        # This is taken care of directly in hard_example_mining
         
        # Anchor and positive are same if only 1 image exists
        # Low probability otherwise
        if nums == 1:
            anchor_name = names[0]
            positive_name = names[0]
        else:
            anchor_name, positive_name = random.sample(names, 2)

        #print("getting images")
        #print("Label:", label, "names:",  anchor_name, positive_name)
        # Negative is picked from labels excluding anchor and new_whale
        if new_whale:
          negative_label = random.choice(list(set(self.labels) ^ set([label, 'new_whale'])))
          # Previously excluded as I did not have this new_whale thing
          negative_label2 = 'new_whale'
          negative_name2 = random.choice(self.dict_train[negative_label2])
          negative_image2, negative_add2 = self.get_image(negative_name2, self.transform_train, negative_label2)
        else:
          negative_label = random.choice(list(set(self.labels) ^ set([label])))
        negative_name = random.choice(self.dict_train[negative_label])

        #print("In the __getitem__")
        #print(anchor_name, label)
        anchor_image, anchor_add = self.get_image(anchor_name, self.transform_train, label)
        positive_image, positive_add = self.get_image(positive_name, self.transform_train, label)
        negative_image,  negative_add = self.get_image(negative_name, self.transform_train, negative_label)

        assert anchor_name != negative_name
        if new_whale:
          return [anchor_image, positive_image, negative_image, negative_image2], \
                 [self.labels_dict[label] + anchor_add, self.labels_dict[label] + positive_add, self.labels_dict[negative_label] + negative_add, self.labels_dict[negative_label2] + negative_add2]
        else:
          return [anchor_image, positive_image, negative_image], \
                 [self.labels_dict[label] + anchor_add, self.labels_dict[label] + positive_add, self.labels_dict[negative_label] + negative_add]


class WhaleTestDataset(Dataset):
    def __init__(self, names, labels=None, mode='test',transform=None):
        super(WhaleTestDataset, self).__init__()
        self.names = names
        self.labels = labels
        self.mode = mode
        self.bbox_dict = self.load_bbox()
        self.labels_dict = self.load_labels()
        self.rle_masks = self.load_mask()
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def get_image(self, name, transform, mode='train'):
        image = cv2.imread('./input/{}/{}'.format(mode, name))
        try:
            mask = do_length_decode(self.rle_masks[name.split('.')[0]]['rle_mask'])
            mask = cv2.resize(mask, image.shape[:2][::-1])
        except:
            mask = cv2.imread('./input/masks/' + name, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros_like(image[:, :, 0])
        if name in self.bbox_dict: 
            x0, y0, x1, y1 = self.bbox_dict[name]
            image = image[int(y0):int(y1), int(x0):int(x1)]
            mask = mask[int(y0):int(y1), int(x0):int(x1)]
        image = transform(image, mask)
        return image

    def load_labels(self):
        label = pd.read_csv('./input/label.csv')
        labelName = label['name'].tolist()
        dict_label = {}
        id = 0
        for name in labelName:
            if name == 'new_whale':
                dict_label[name] = NUM_CLASSES
                continue
            dict_label[name] = id
            id += 1
        return dict_label

    def load_mask(self):
        rle_masks = pd.read_csv('./input/model_50A_slim_ensemble.csv')
        rle_masks = rle_masks[rle_masks['rle_mask'].isnull() == False]
        rle_masks.index = rle_masks['id']
        del rle_masks['id']
        rle_masks = rle_masks.to_dict('index')
        return rle_masks

    def load_bbox(self):
        print('loading bbox...')
        bbox = pd.read_csv('./input/bboxs.csv')
        Images = bbox['Image'].tolist()
        x0s = bbox['x0'].tolist()
        y0s = bbox['y0'].tolist()
        x1s = bbox['x1'].tolist()
        y1s = bbox['y1'].tolist()
        bbox_dict = {}
        for Image, x0, y0, x1, y1 in zip(Images, x0s, y0s, x1s, y1s):
            bbox_dict[Image] = [x0, y0, x1, y1]
        return bbox_dict
    def __getitem__(self, index):
        if self.mode in ['test']:
            name = self.names[index]
            image = self.get_image(name, self.transform, mode='test')
            return image, name
        elif self.mode in ['valid', 'train', 'embed']:
            name = self.names[index]
            label = self.labels_dict[self.labels[index]]
            if self.mode == 'embed':
                image = self.get_image(name, self.transform, mode='test')
            else:
                image = self.get_image(name, self.transform)
            #print(label)
            return image, label, name




