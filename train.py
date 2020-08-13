import datetime
import os

from timeit import default_timer as timer
from dataSet import *
from models import *
import torch
import time
from utils import *
from torch.nn.parallel.data_parallel import data_parallel
import numpy as np
import pandas as pd
# from torchsummary import summary

# To stop cv2/dataloader deadlock type situation
cv2.setNumThreads(0)

# Some global vars for convenience
width = 512
height = 256
#classes = 5004
#classes = 2904
classes = 839
# Are masks included with the images through the model
withMask = True 
# Used when testing a feature (like when first having new_whale in training set)
test = False 
# 9999 These two have been changed for training the old models
newWhale = True
oldModel = False

def train_collate(batch):

    batch_size = len(batch)
    images = []
    labels = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            labels.extend(batch[b][1])
    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))
    return images, labels

def valid_collate(batch):

    batch_size = len(batch)
    images = []
    labels = []
    names = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            labels.append(batch[b][1])
            names.append(batch[b][2])
    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))
    return images, labels, names

def transform_train(image, mask, label):
    add_ = 0

    image = cv2.resize(image, (512, 256))
    mask = cv2.resize(mask, (512, 256))
    mask = mask[:,:, None]
    
    # This is needed for the flipping. In case flipping happens,
    # we want both to be together. We separate them again after that
    image = np.concatenate([image, mask], 2)
    # if 0:
    #     if random.random() < 0.5:
    #         image = bgr_to_gray(image)

    # Apply some data augmentation
    # Flip
    if 1:
        if random.random() < 0.5:
            image = np.fliplr(image)
            # Create a new id when flipped l->r, if id is known (not 'new_whale')
            if not label == 'new_whale':
                add_ += classes
        image, mask = image[:,:,:3], image[:,:, 3]
    
    # Rotate
    if random.random() < 0.5:
        image, mask = random_angle_rotate(image, mask, angles=(-25, 25))
    
    # Noise
    if random.random() < 0.5:
        index = random.randint(0, 1)
        if index == 0:
            image = do_speckle_noise(image, sigma=0.1)
        elif index == 1:
            image = do_gaussian_noise(image, sigma=0.1)
    
    # Lighting ?
    if random.random() < 0.5:
        index = random.randint(0, 3)
        if index == 0:
            image = do_brightness_shift(image,0.1)
        elif index == 1:
            image = do_gamma(image, 1)
        elif index == 2:
            image = do_clahe(image)
        elif index == 3:
            image = do_brightness_multiply(image)
    
    # Erase part of the image ?
    if 1:
        image, mask = random_erase(image,mask, p=0.5)
    # Shift part of the image ?
    if 1:
        image, mask = random_shift(image,mask, p=0.5)
    # Scale the image ?
    if 1:
        image, mask = random_scale(image,mask, p=0.5)
    # todo data augment (<- what did he mean?)
    if 1:
        if random.random() < 0.5:
            mask[...] = 0

    # Concatenate image and mask for the input,
    # convert to the right type, and normalize
    mask = mask[:, :, None]
    if withMask:
      image = np.concatenate([image, mask], 2)
    image = np.transpose(image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    return image, add_

def transform_valid(image, mask):
    images = []
    image = cv2.resize(image, (512, 256))
    mask = cv2.resize(mask, (512, 256))
    mask = mask[:, :, None]
    if withMask:
      image = np.concatenate([image, mask], 2)
    raw_image = image.copy()

    image = np.transpose(raw_image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    images.append(image)

    image = np.fliplr(raw_image)
    image = np.transpose(image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    images.append(image)
    return images

def transform_embed(image, mask):
    image = cv2.resize(image, (512, 256))
    mask = cv2.resize(mask, (512, 256))
    mask = mask[:, :, None]
    if withMask:
      image = np.concatenate([image, mask], 2)
    raw_image = image.copy()

    image = np.transpose(raw_image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    return image

def eval(model, dataLoader_valid):
    with torch.no_grad():
        model.eval()
        model.mode = 'valid'
        valid_loss, index_valid= 0, 0
        all_results = []
        all_labels = []
        for valid_data in dataLoader_valid:
            images, labels, names = valid_data
            images = images.cuda()
            labels = labels.cuda().long()
            feature, local_feat, results = data_parallel(model, images)
            model.getLoss(feature[::2], local_feat[::2], results[::2], labels)
            results = torch.sigmoid(results)
            # Combine the guesses for the results: since the outputs
            # are arranged so that an image and its flipped counterpart
            # are right after one another. And the flipped version's
            # correct classification is the original's ID + classes.
            # So, we add the 2 (out of 4) quadrants that contain the
            # correct intersection and ignore the rest (guess it doesn't matter
            # if it's wrong anyways?... )
            # (This is not necessarily an accurate representation
            # of the overall accuracy. But it does boost the accuracy...
            # Seems the network can sometimes better recognize an image as
            # an individual when its flipped.. interesting)
            results_zeros = (results[::2, :classes] + results[1::2, classes:])/2
            all_results.append(results_zeros)
            all_labels.append(labels)
            b = len(labels)
            valid_loss += model.loss.data.cpu().numpy() * b
            index_valid += b
        all_results = torch.cat(all_results, 0)
        all_labels = torch.cat(all_labels, 0)
        map10s, top1s, top5s, top10s = [], [], [], []
        if 1:
            ts = np.linspace(0.1, 0.9, 9)
            for t in ts:
                # Guess that the id is 'new_whale' if nothing else has a high enough probability
                # We don't know what a good threshold is here, so test out a bunch
                results_t = torch.cat([all_results, torch.ones_like(all_results[:, :1]).float().cuda() * t], 1)
                # The guess for new_whale is appended to the end. So the last id is for 'new_whale'
                all_labels[all_labels == classes * 2] = classes
                top1_, top5_, top10_ = accuracy(results_t, all_labels, topk=(1,5,10))
                map10_ = mapk(all_labels, results_t, k=5)
                map10s.append(map10_)
                top1s.append(top1_)
                top5s.append(top5_)
                top10s.append(top10_)
            map10 = max(map10s)
            i_max = map10s.index(map10)
            top1 = top1s[i_max]
            top5 = top5s[i_max]
            top10 = top10s[i_max]
            best_t = ts[i_max]

        valid_loss /= index_valid
        #return valid_loss, top1, top5, map5, best_t
        return valid_loss, top1, top5, top10, map10, best_t

def eval_embed(model, dataloader_valid, embeddings, ind_train_toID):
    # Setup torch and model for evaluation and efficiency
    with torch.no_grad():
        model.eval()
        # TODO: Test this. Do not think this is necessary
        #model.mode = 'valid'
        #valid_loss, index_valid= 0, 0
        all_feats = []
        all_local = []
        # Embedded
        all_results_e = []
        all_labels = []
        
        # Run dataset through model and get accuracy
        for data in dataloader_valid:
            images, labels, _ = data
            images = images.cuda()
            labels = labels.cuda().long()
            features, local_feat, outs = data_parallel(model, images)
            # NEW loss
            #model.getLoss(features[::2], local_feat[::2], outs[::2], labels)

            """
            print(type(features), features.dtype)
            print(features.size())
            print(type(embeddings), embeddings.dtype)
            print(embeddings.size())
            """
            # Get distances to embeddings
            # dist_mat: pytorch Variable, with shape [N, M]
            dist_mat = euclidean_dist(features.cuda(), embeddings.cuda())
            #dist_mat = euclidean_dist(outs.cuda(), embeddings.cuda())
            # TODO: trying new!
            #dist_mat = features.cuda().mm(embeddings.cuda().t())
            #dist_mat = outs.mm(embeddings.cuda().t())
            
            """
            print(dist_mat)
            print(dist_mat.size())
            """
            flip_lr = True
            if flip_lr:
                # TODO: Test and decide which to do
                # Will do nearest neighbors first then try "center".
                # OR
                # Right now, I am trying to compare my image features to embeddings of every
                # single training image. But what I need is to compare them to embeddings
                # of the LABELS... So, I need to get a "center" vector for each label and 
                # compare my features to those instead...
                known = embeddings.size()[0]//2
                assert known == ind_train_toID.size()[0]
                # TODO: TRYING THIS OUT.. THINK WAS A MISTAKE...???
                # Labels are no longer 1...n, then flipped. But 1,1f,...n,nf right next
                # since I am not using model outputs but batch input to get embeddings
                #dist_mat = (dist_mat[::2, :known] + dist_mat[1::2, known:])/2
                dist_mat = (dist_mat[::2, ::2] + dist_mat[1::2, 1::2])/2

            # Currently, what we have is a mapping of distances to images' index but we want
            # a mapping of distances to the ID that corresponds to those images.
            # Instead of creating a new accuracy fcn, if I change the range of the 
            # distance to [0..1], and then do 1 - distances, then I can just use the rest of
            # the set up just like if it were classification. Accuracy fcn was just
            # modified to convert the indices to IDs
            
            # Trying to make sure to get the resuts to be like probabilities
            # so acuracy function can be used as well
            # TODO: remove this to see how it influences accuracy.
            # TODO: JUST REPLACED THIS AND NEXT LINE WITH NEW WAY AS TEST
            #dist_mat = torch.sigmoid(dist_mat)

            #all_results_e.append(1 - dist_mat)
            all_results_e.append(dist_mat)
            all_labels.append(labels)
            all_feats.append(features[::2])
            all_local.append(local_feat[::2])
            #b = len(labels)
            # NEW loss
            #valid_loss += model.loss.data.cpu().numpy() * b
            #index_valid += b
        all_results_pre = torch.cat(all_results_e, 0)
        all_labels = torch.cat(all_labels, 0)
        all_feats = torch.cat(all_feats, 0)
        all_local = torch.cat(all_local, 0)
        #print("In eval:")
        print("Min and Max distances seen: {} {}".format(all_results_pre.min(), all_results_pre.max()))
        
        #""" 
        #print("Some results visualized:")
        #see, _ = all_results_e.topk(5, 1, True, True)
        #print(see[:20])
        min_dist = 0.8
        #min_dist = -0.3 
        #min_dist = 0 #1900
        #min_dist = 2000
        min_found = all_results_pre.min()
        if min_found < min_dist:
            print("#"*20)
            print("Lower min found: {}".format(min_found))
            min_dist = min_found
        results3 = all_results_pre - min_dist
        # Don't necessarily need this second part if max_dist is close to 1
        max_dist = 0.9    # 1.7 - 0.8 OR # 0.6 + 0.3
        #max_dist = 200   #200-0  # ??? 8400-1900=6500 
        #max_dist = 6200   # 8200-2000 
        max_found = results3.max()
        if max_found > max_dist:
            print("#"*20)
            print("Higher max found: {}".format(max_found))
            max_dist = max_found
        results2 = results3 / max_dist
        all_results_e = 1 - results2
        #"""
        #all_results_e = all_results_pre
    # End of with

    # NEW loss
    #valid_loss /= index_valid
    model.getLoss(all_feats, all_local, all_results_e, all_labels, ind_train_toID)
    valid_loss = model.loss.data.cpu().numpy()

    # TODO: implement giving each 'new_whale' their own ID so accuracy metric is more
    # accurate.. (as done on paper w/ explanation to Hilal) if min occ. to include is 1.
    # This won't relly be helpful with 'new_whale's as I can't test for re-identification
    # but I will def need this to include 'new_whale's with more than 1 occurence 
    # to test that the embedding approach works
    map10s, top1s, top5s, top10s = [], [], [], []
    new_res, known_res = [], []
    # Adding the mapping for the 'label_map', 
    # as 'new_whale' threshold added to results_t is not
    # a part of the training set data
    label_map = torch.cat([ind_train_toID, torch.Tensor([CLASSES]).cuda().long()], 0)
    if 1:
        ts = np.linspace(0.1, 0.9, 9)
        for t in ts:
            # Guess that the id is 'new_whale' if nothing else has a high enough probability
            # We don't know what a good threshold is here, so test out a bunch
            results_t = torch.cat([all_results_e, torch.ones_like(all_results_e[:, :1]).float().cuda() * t], 1)
            # The guess for new_whale is appended to the end. So the last id is for 'new_whale'
            # We convert the 'new_whale' ID from its original state: 'the last ID after all the flipped IDs'
            all_labels[all_labels == CLASSES * 2] = CLASSES
            #new_whale, known, top1_, top5_ = accuracy(results_t, all_labels, label_map=label_map, sep=CLASSES)
            # MOD 9
            new_whale, known, top1_, top5_, top10_ = accuracy(results_t, all_labels, topk=(1,5,10), label_map=label_map, sep=CLASSES)
            #map5_ = mapk(all_labels, results_t, k=5, label_map=label_map)
            map10_ = mapk(all_labels, results_t, k=10, label_map=label_map)
            """
            print("t:", t, "map:", map10_)
            print("e: known: {}, new: {}".format(known, new_whale))
            #print("top1: {}, top5: {}".format(top1_, top5_))
            print("top1: {}, top5: {}, top10: {}".format(top1_, top5_, top10_))
            """
            map10s.append(map10_)
            new_res.append(new_whale)
            known_res.append(known)
            top1s.append(top1_)
            top5s.append(top5_)
            top10s.append(top10_)
        map10 = max(map10s)
        i_max = map10s.index(map10)
        new_b = new_res[i_max]
        known_b = known_res[i_max]
        top1 = top1s[i_max]
        top5 = top5s[i_max]
        top10 = top10s[i_max]
        best_t = ts[i_max]
        # TODO: Output the results when separated by known vs. 'new_whale' as well
        print("Best in eval")
        print("e: known: {}, new: {}".format(known_b, new_b))
        print("top1: {}, top5: {}".format(top1, top5))

    #return valid_loss, top1, top5, map5, best_t
    return valid_loss, top1, top5, top10, map10, best_t

def get_features(dataloader_embed, model, num_classes):
    # Acquire the features to use for embedding #, if necessary
    #if embed or top1 > max_valid:
    out_ids = []
    out_features = []
    for store in dataloader_embed:
        images, labels, names = store
        images = images.cuda()
        labels = labels.cuda().long()
        #print(labels)
        #print(names)
        if not(test):
          global_feat, local_feat, results = data_parallel(model,images)
        else:
          global_feat, local_feat, results = model.forward(images)
        #print(global_feat.shape)
        out_ids.extend(labels.cpu().long().numpy())
        out_features.extend(global_feat.data.cpu().numpy())
        #out_features.extend(results.data.cpu().numpy())
    #exit()

    # Modify the features to remove all 'new_whale's, 
    # as those are not proper IDs, ie. different whales 
    # are grouped up with this 'ID' if they have no known ID
    ids, feats = [], []
    for ind, x in enumerate(out_ids):
        if int(x) == num_classes:
            continue
        ids.append(x)
        y = out_features[ind*2]
        y2 = out_features[ind*2+1]
        feats.append(y)
        feats.append(y2)
    return ids, feats

def train(debug, freeze=False, fold_index=1, model_name='seresnext50',min_num_class=10, checkpoint_start=0, lr=3e-4, batch_size=36, num_classes=10008, embed=False):
    import ipdb
      
    device = torch.device('cuda')
    if withMask:
      #model = model_whale(num_classes=num_classes, inchannels=4, model_name=model_name).cuda()
      model = model_whale(num_classes=num_classes, inchannels=4, model_name=model_name).to(device)
    else:
      model = model_whale(num_classes=num_classes, inchannels=3, model_name=model_name).cuda()
    if debug:
        print(model)
        print("\n\n\n\n\n\n\n\n\n")
        #print("     999999999          ")
        #summary(model.cuda(), (3, width, height))
    
    i = 0
    # We want more valid/saving for embed as the training times are much, much longer
    if embed:
        # Changed this since this method is much slower, helps to keep track
        iter_smooth = 10
        #iter_valid = 50
        # Valid and save should be the same so I know the state at the time of save!
        iter_valid = 20
        iter_save = 20
    else:
        iter_smooth = 50
        iter_valid = 200
        iter_save = 200
    epoch = 0
    if freeze:
        model.freeze()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  betas=(0.9, 0.99), weight_decay=0.0002)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0002)
    resultDir = './result/{}_{}'.format(model_name, fold_index)
    ImageDir = resultDir + '/image'
    checkPoint = os.path.join(resultDir, 'checkpoint')
    os.makedirs(checkPoint, exist_ok=True)
    os.makedirs(ImageDir, exist_ok=True)
    log = Logger()
    log.open(os.path.join(resultDir, 'log_train.txt'), mode= 'a')
    log.write(' start_time :{} \n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    log.write(' batch_size :{} \n'.format(batch_size))
    
    # Image,Id
    # if debug: ipdb.set_trace(context=5)
    
    data_train = pd.read_csv('./input/train_split_{}.csv'.format(fold_index))
    names_train = data_train['Image'].tolist()
    labels_train = data_train['Id'].tolist()
    data_valid = pd.read_csv('./input/valid_split_{}.csv'.format(fold_index))
    names_valid = data_valid['Image'].tolist()
    labels_valid = data_valid['Id'].tolist()
    num_data = len(names_train)
    num_vld = len(names_valid)
    ind1 = len(set(labels_train))
    ind2 = len(set(labels_valid))
    print()
    print("Initial incoming image analysis")
    log.write("\nLooking at {} total images.\n".format(num_data+num_vld))
    log.write("{} train and {} validation images\n".format(num_data, num_vld))
    log.write("{} train and {} valid individuals\n".format(ind1, ind2))

    # Get the dataset for training
    if not(test):
      # For retraining with older model, unfinalized ideas
      if oldModel:
          dst_train = WhaleDataset(names_train, labels_train, mode='train', transform_train=transform_train, min_num_classes=min_num_class, newWhale=newWhale)
          dataloader_train = DataLoader(dst_train, shuffle=True, drop_last=True, batch_size=batch_size, num_workers=16, collate_fn=train_collate)
          # No dataloader for use to store embeddings
      else:
          dst_train = WhaleDataset(names_train, labels_train, mode='train', transform_train=transform_train, min_num_classes=min_num_class)
          dataloader_train = DataLoader(dst_train, shuffle=True, drop_last=True, batch_size=batch_size, num_workers=16, collate_fn=train_collate)
          
          # Create the dataloader for use to store embeddings
          dst_embed = WhaleTestDataset(names_train, labels_train, mode='train', transform=transform_valid)
          dataloader_embed = DataLoader(dst_embed, shuffle=False, drop_last=False, batch_size=batch_size, num_workers=16, collate_fn=valid_collate)
        # num_workers was previously 4 TODO: TEST THIS!!!
    
    else:
      dst_train = WhaleDataset(names_train, labels_train, mode='train', transform_train=transform_train, min_num_classes=min_num_class, newWhale=newWhale)
      # Change drop_last back to True if there is an issue... 
      # Thought this would cause an error but apparently not
      dataloader_train = DataLoader(dst_train, shuffle=True, drop_last=False, batch_size=batch_size, num_workers=0, collate_fn=train_collate)
      
      # Create the dataloader for use to store embeddings
      dst_embed = WhaleTestDataset(names_train, labels_train, mode='train', transform=transform_embed)
      dataloader_embed = DataLoader(dst_embed, shuffle=False, drop_last=False, batch_size=3, num_workers=0)

    # Get the dataset for validation
    dst_valid = WhaleTestDataset(names_valid, labels_valid, mode='valid', transform=transform_valid)
    if not(test):
      dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=batch_size * 2, num_workers=8, collate_fn=valid_collate)
    # num_workers was previously 4 TODO: TEST THIS!!
    else:
      dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=1, num_workers=0, collate_fn=valid_collate)
    #ipdb.set_trace(context=5)
    
    print()
    print("Analysis of images to be used")
    print("Train & valid image analysis:")
    print("Number of training images with at least {} instances: {}".format(min_num_class, dst_train.num_images)) 
    print("Number of validation images with at least {} instances: {}".format(0, dst_valid.__len__()))
    
    # Right now it is important for everything in valid to be in train
    # (this should be changed later on) 
    trn = set(dst_train.labels)
    vld = set(dst_valid.labels)
    extra = vld-trn
    i2 = len(vld)
    i1 = i2 - len(extra)

    print("There are {} individuals found in train".format(len(trn)))
    print("{} out of {} ids in valid are found in train".format(i1, i2))
    if (i1 < i2):
      print("The missing ones are: {}".format(extra))
    print()

    train_loss = 0.0
    valid_loss = 0.0
    #top1, top5, map5 = 0, 0, 0
    top1, top5, top10, map10 = 0, 0, 0, 0
    top1_train, top5_train, map10_train = 0, 0, 0
    #top1_batch, top5_batch, map5_batch = 0, 0, 0
    top1_batch, top5_batch, top10_batch, map10_batch = 0, 0, 0, 0

    batch_loss = 0.0
    train_loss_sum = 0
    train_top1_sum = 0
    train_map10_sum = 0
    sum_ = 0
    skips = []
    if not checkpoint_start == 0:
        log.write('Starting from iter {}, l_rate = {} \n'.format(checkpoint_start, lr))
        log.write('freeze={}, batch_size={}, min_num_class={} \n\n'.format(freeze,batch_size, min_num_class))
        model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (checkpoint_start)),skip=skips)
        ckp = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (checkpoint_start)))
        optimizer.load_state_dict(ckp['optimizer'])
        adjust_learning_rate(optimizer, lr)
        i = checkpoint_start
        epoch = ckp['epoch']
    #log.write(
    #        ' rate     iter   epoch  | valid   top@1    top@5    map@5  | '
    #        'train    top@1    top@5    map@5 |'
    #        ' batch    top@1    top@5    map@5 |  time          \n')
    log.write(
	    'format: \n'
            'rate   iter  k  epoch | valid   top@1   top@5   map@5   best_t |'
            ' train   top@1   map@5 |'
            ' batch   top@1   map@5 | time \n')
    log.write(
            '--------------------------------------------------------------------------------------------------------------------------------------\n')
    start = timer()

    start_epoch = epoch
    best_t = 0
    cycle_epoch = 0
    curMax = 0
    max_valid = 0
    if not(oldModel):
        # Load the previously stored features with max validation accuracy
        # This features is to stop runs from overwriting previously gotten, good
        # features because the system has no concept of what was previously achieved
        if embed:
            acc_file = "maxValAcc_{}_{}_embed.txt".format(model_name, fold_index)
        else:
            acc_file = "maxValAcc_{}_{}.txt".format(model_name, fold_index)
        # To avoid overwriting previously gotten results that were better
        try:
            max_valid = pd.read_csv(acc_file)['0'][0]
            print("Found max valid of:", max_valid)
        except:
            print("No existing max valid found. Creating one.")
            handle = open(acc_file, "w+")
            handle.close()
    
    #from ipdb import launch_ipdb_on_exception
    #ipdb.set_trace(context=5)
    # with launch_ipdb_on_exception():
    if True:
      while i < 10000000:
          print("\nAt iteration:", i)
          for data in dataloader_train:
              #print("\nIteration:",i)
              # Starting model learning process
              epoch = start_epoch + (i - checkpoint_start) * 4 * batch_size/num_data
              
              # Temporarily addded to test convergence when running with 'old' embeddings
              #if i % iter_valid == 0:
              if embed:
                  ids, feats = get_features(dataloader_embed, model, num_classes)
                  
              # Validation phase
              if i % iter_valid == 0:
                  if not(embed):
                      valid_loss, top1, top5, top10, map10, best_t = \
                          eval(model, dataloader_valid)

                  # Run embedding validation function
                  if embed:
                      valid_loss, top1, top5, top10, map10, best_t = \
                          eval_embed(model, dataloader_valid, \
                          torch.Tensor(feats).float(), torch.Tensor(ids).cuda().long())
                      
                  if top1 > curMax:
                      curMax = top1
                      print("New max valid for this run")
                  # Store the features of the best model version for embedding
                  if top1 > max_valid and not(oldModel):
                      if not(embed):
                          ids, feats = get_features(dataloader_embed, model, num_classes)
                      max_valid = top1
                      print("Saving current features. Best valid acc. found.")
                      # Save data
                      # They are not being saved together as they are not the same
                      # length. Could be though. 
                      addition = ""
                      if embed:
                          addition = "2"
                      outfile = "train_ids_{}{}.csv".format(fold_index, addition)
                      outfile2 = "train_features_{}{}.csv".format(fold_index, addition)
                      df1 = pd.DataFrame(ids)
                      df2 = pd.DataFrame(feats)
                      # Keep track of id, vector and some info about where this
                      # was gotten (model, fold, iteration?)
                      df1.to_csv(outfile, index=None)
                      df2.to_csv(outfile2, index=None)
                      pd.DataFrame([max_valid.cpu().numpy()]).to_csv(acc_file, index=None)
                      #np.savetxt(outfile, out_ids)
                      #np.savetxt(outfile2, out_features)

                  # Output some statistics
                  print('\r', end='', flush=True)
                  log.write(
                      '%0.5f %5.2f v %5.2f  |'
                      ' %0.3f    %0.3f    %0.3f    %0.3f    %0.4f    %0.4f | %0.3f    %0.3f    %0.3f | %0.3f     %0.3f    %0.3f | %s\n' % ( \
                          lr, i / 1000, epoch,
                          valid_loss, top1, top5, top10, map10, best_t,
                          train_loss, top1_train, map10_train,
                          batch_loss, top1_batch, map10_batch,
                          time_to_str((timer() - start) / 60)))
                  # Don't know if this played a role but potentially slows it down..
                  #time.sleep(0.01)
              # End valid if statement
  
              #print("check1")
              # Save the state of the model and some attributes for possible reuse
              if i % iter_save == 0 and not i == checkpoint_start:
                  torch.save(model.state_dict(), resultDir + '/checkpoint/%08d_model.pth' % (i))
                  torch.save({
                      'optimizer': optimizer.state_dict(),
                      'iter': i,
                      'epoch': epoch,
                      'best_t':best_t,
                  }, resultDir + '/checkpoint/%08d_optimizer.pth' % (i))

              # This doesn't actually do any training but sets up the 
              # modules in the model so that they are ready for training
              model.train()

              # Get training results
              model.mode = 'train'
              images, labels = data
              images = images.cuda()
              labels = labels.cuda().long()
              global_feat, local_feat, results = data_parallel(model,images)
              if embed:
                  embeddings = torch.Tensor(feats).float()
                  # Create a modified target ids with the mirrored training images included
                  dist_mat  = euclidean_dist(global_feat, embeddings.cuda())
                  #dist_mat  = euclidean_dist(results, embeddings.cuda())
                  # TODO: trying new!
                  #dist_mat = global_feat.mm(embeddings.cuda().t())
                  #dist_mat = results.mm(embeddings.cuda().t())
                  # Here we are trying to make sure our 1-sigmoid(x) scheme is appropriate
                  print("Min and Max distances seen: {} {}".format(dist_mat.min(), dist_mat.max()))
                  #results = 1 - torch.sigmoid(dist_mat)
                  # Upper and lower bounds for distance results from above
                  min_dist = 0.8
                  #min_dist = -0.3
                  #min_dist = 0 # 1900
                  #min_dist = 2000
                  min_found = dist_mat.min()
                  if min_found < min_dist:
                      print("#"*20)
                      print("Lower min found: {}".format(min_found))
                      min_dist = min_found
                  results3 = dist_mat - min_dist
                  # TODO: ATTEMPT THIS WITH TORCH.NORM OR SOMETHING AUTOMATIC
                  # AND COMPARE RESULTS AS WELL
                  # Don't necessarily need this second part if max_dist is close to 1
                  max_dist = 0.9    # 1.7 - 0.8
                  #max_dist = 0.9    # 0.6 + 0.3
                  #max_dist = 200   #200-0  # ??? 8400-1900=6500 
                  #max_dist = 6200   # 8200-2000 
                  max_found = results3.max()
                  if max_found > max_dist:
                      print("#"*20)
                      print("Higher max found: {}".format(max_found))
                      max_dist = max_found
                  results2 = results3 / max_dist
                  results = 1 - results2
                  #"""
                  #results = dist_mat

                  # For training, we do need to add the second, mirrored label
                  # to the ids as those are considered different classes 
                  # and not combined as they are in validation and testing
                  ids_ = []
                  for elem in ids: ids_.extend([elem, elem+classes])
                  ind_train_toID = torch.Tensor(ids_).cuda().long()
                  #model.getLoss(global_feat, local_feat, results, labels, ind_train_toID)
                  model.getLoss(global_feat, local_feat, None, labels)
              else:
                  model.getLoss(global_feat, local_feat, results, labels)
              batch_loss = model.loss
               
              #print("check3")
              # Backpropagation and accuracy measurement
              optimizer.zero_grad()
              batch_loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
              optimizer.step()
              #print("check4")
              if embed:
                  results = torch.cat([results, torch.ones_like(results[:, :1]).float().cuda() * 0.5], 1)
                  # This makes sure that there are no images classified as 'new_whale'
                  # in the set of seen images. They should ALL be known (ie. have an ID)
                  assert ind_train_toID.ne(num_classes).all()
                  label_map = torch.cat([ind_train_toID, torch.Tensor([num_classes]).cuda().long()], 0)
                  # Mode here was to be potentially used if needed to remove embeddings
                  # of the same image for the training set. 
                  # Initially thought it screwed the accuracy.. not used tho
                  new_whale, known, top1_batch, top5_batch, top10_batch = accuracy(results, labels, topk=(1,5,10), label_map=label_map, sep=num_classes, mode="train")
                  #map5_batch = mapk(labels, results, k=5, label_map=label_map)
                  map10_batch = mapk(labels, results, k=10, label_map=label_map)
                  #print("Just ran accuracy")
                  print("In train")
                  print("e: known: {}, new: {}, top5: {}, top10: {}".format(known, new_whale, top5_batch, top10_batch))
              else:
                  results = torch.cat([torch.sigmoid(results), torch.ones_like(results[:, :1]).float().cuda() * 0.5], 1)
                  #top1_batch = accuracy(results, labels, topk=(1,))[0]
                  new_whale, known, top1_batch, top5_batch, top10_batch = accuracy(results, labels, topk=(1,5,10), sep=num_classes)
                  map10_batch = mapk(labels, results, k=5)

              #print("check5")
              # Nothing after this should need to change
              batch_loss = batch_loss.data.cpu().numpy()
              sum_ += 1
              train_loss_sum += batch_loss
              train_top1_sum += top1_batch
              train_map10_sum += map10_batch

              # Aggregrate the training data over the last 50 iterations 
              # and show the average loss, accuracy & map
              if (i + 1) % iter_smooth == 0:
                  print("\nSmoothing after {}".format(iter_smooth))
                  train_loss = train_loss_sum/sum_
                  top1_train = train_top1_sum/sum_
                  map10_train = train_map10_sum/sum_
                  train_loss_sum = 0
                  train_top1_sum = 0
                  train_map10_sum = 0
                  sum_ = 0

                  if 1:
                  #if embed:
                      print("e: known: {}, new: {}, top5: {}".format(known, new_whale, top5_batch))
                  # This was indented as I do not care to see this output so often...
                  print('%0.5f %5.2f l %5.2f  | %0.3f    %0.3f    %0.3f    %0.4f    %0.4f | %0.3f    %0.3f    %0.3f | %0.3f     %0.3f    %0.3f | %s  %d %d\n' % ( \
                      lr, i / 1000, epoch,
                      valid_loss, top1, top5, map10, best_t,
                      train_loss, top1_train, map10_train,
                      batch_loss, top1_batch, map10_batch,
                      time_to_str((timer() - start) / 60), checkpoint_start, i)
                  , end='', flush=True)
              i += 1
             
          pass


if __name__ == '__main__':
    #torch.multiprocesssing.set_start_method('spawn')

    if 1:
        # Light command-line arguments check 
        try:
          fold_index = sys.argv[1]
          checkpoint_start = int(sys.argv[2])
          lr = float(sys.argv[3])
          embed = bool(int(sys.argv[4]))
        except:
          print("Input: ", sys.argv)
          print("Usage: {} fold_index checkpoint_start lr embed".format(sys.argv[0]))
          print("Default: new_flukes 0 3e-4 1")
          print("embed must be either 1(True) or 0(False)")
          exit(1)
          # fold_index = "flukes10"
          # checkpoint_start = 0
          # lr = 3e-4
          # embed = True
       
        # Other debugging info
        # 9999
        debug = True
        #try:
        #  debug = sys.argv[5]
        #except:
        #  debug = False
       
        # Other necessary variables
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,5'
        freeze = False
        # this variable should NEVER change! change the input fold_index instead
        min_num_class = 0
        # 24 ran out of memory with 2 32gb GPUs
        # but worked with 6 and 360min runtime
        if embed:
            batch_size = 24
        else:
            batch_size = 12
        num_classes = classes * 2

        origin_model = True
        if origin_model:
          model_name = 'senet154'
        else:
          model_name = 'seresnet101'
        
        print("Some params:")
        print("Fold_index: {}, Check_start: {}, Learning rate: {}, Min # of Occ.: {}, Embedding: {}".format(fold_index, checkpoint_start, lr, min_num_class, embed))
        print("!!! Sanity check !!!")
        print("Input: ", sys.argv)
        print()
        print("Chosen model: {}".format(model_name))
        print()
        
        train(debug, freeze, fold_index, model_name, min_num_class, checkpoint_start, lr, batch_size, num_classes, embed)
