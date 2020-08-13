import os
import torch
import shutil

import pandas as pd
import numpy as np
from tqdm import tqdm
from models import *
from dataSet import *
from utils.metric import accuracy, mapk
from models.triplet_loss import euclidean_dist

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# For ease of debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
num_TTA = 2

# To stop cv2/dataloader deadlock type situation
cv2.setNumThreads(0)

#CLASSES = 5004
# 9999 Changed this
#CLASSES = 2904
CLASSES = 839

def train_collate(batch):

    batch_size = len(batch)
    images = []
    names = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            names.append(batch[b][1])
    images = torch.stack(images, 0)
    return images, names


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

# Only difference is that it only returns one image (no flipping)
def transform_embed(image, mask):
    raw_image = cv2.resize(image, (512, 256))
    raw_mask = cv2.resize(mask, (512, 256))
    raw_mask = raw_mask[:, :, None]
    raw_image = np.concatenate([raw_image, raw_mask], 2)

    image = np.transpose(raw_image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    return image


def test(checkPoint_start=0, fold_index=1, model_name='senet154'):
    names_test = os.listdir('./input/test')
    batch_size = 16
    dst_test = WhaleTestDataset(names_test, mode='test', transform=transform)
    dataloader_test = DataLoader(dst_test, batch_size=batch_size, num_workers=8, collate_fn=train_collate)
    # Get mapping of model IDS to labels for output
    label_id = dst_test.labels_dict
    id_label = {v:k for k, v in label_id.items()}
    id_label[5004] = 'new_whale'
    model = model_whale(num_classes=5004 * 2, inchannels=4, model_name=model_name).cuda()
    resultDir = './result/{}_{}'.format(model_name, fold_index)
    checkPoint = os.path.join(resultDir, 'checkpoint')

    npy_dir = resultDir + '/out_{}'.format(checkPoint_start)
    os.makedirs(npy_dir, exist_ok=True)
    if not checkPoint_start == 0:
        model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start)),skip=[])
        ckp = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (checkPoint_start)))
        best_t = ckp['best_t']
        print('\nbest_t:', best_t,'\n')
    labelstrs = []
    allnames = []
    with torch.no_grad():
        model.eval()
        for data in tqdm(dataloader_test):
            images, names = data
            images = images.cuda()
            _, _, outs = model(images)
            outs = torch.sigmoid(outs)
            outs_zero = (outs[::2, :5004] + outs[1::2, 5004:])/2
            outs = outs_zero
            for out, name in zip(outs, names):
                out = torch.cat([out, torch.ones(1).cuda()*best_t], 0)
                out = out.data.cpu().numpy()
                np.save(os.path.join(npy_dir, '{}.npy'.format(name)), out)
                top5 = out.argsort()[-5:][::-1]
                str_top5 = ''
                for t in top5:
                    str_top5 += '{} '.format(id_label[t])
                str_top5 = str_top5[:-1]
                allnames.append(name)
                labelstrs.append(str_top5)
    pd.DataFrame({'Image': allnames,'Id': labelstrs}).to_csv('test_{}_sub_fold{}.csv'.format(model_name, fold_index), index=None)

def getLabels(features, embeddings):
    """
    Given input features and embeddings of the training set, calculate the probable
    labels.

    Args:
      features: pytorch LongTensor, with shape [N, 2048]
      embeddings: pytorch LongTensor, with shape [M, 2048]

    Returns:
      labels: pytorch Variable, with shape [N, M]
    """
    #dist_mat = euclidean_dist(features, embeddings)
    return NotImplemented

def getIDCenters(embeddings, labels):
    """
    Given input embeddings of the training set and their associated labels, calculate 
    the centers of all the existing labels as an average of their image embeddings..

    Args:
      embeddings: pytorch LongTensor, with shape [M, 2048]
      labels: pytorch LongTensor, with shape [N]

    Returns:
      centers: pytorch Variable, with shape [N, 2048]
    """
    return NotImplemented

def run_classifier(model, dataloader_embed, best_t):
    with torch.no_grad():
        model.eval()
        all_results = []
        all_labels = []
        
        # Run dataset through model and get accuracy
        for data in dataloader_embed:
            images, labels, names = data
            images = images.cuda()
            labels = labels.cuda().long()
            features, local_feats, outs = model(images)

            # Get loss just to output
            # Get labels through usual method
            outs = torch.sigmoid(outs)
            # This test assumes this image can sometimes be better recognized as its flipped
            # variant... Interesting
            if flip_lr:
              outs_zero = (outs[::2, :CLASSES] + outs[1::2, CLASSES:])/2
              outs = outs_zero

            all_results.append(outs)
            all_labels.append(labels)
        all_results = torch.cat(all_results, 0)
        all_labels = torch.cat(all_labels, 0)
        # These two accuracies should be the same-ish or lower
        # since 'new_whale' id hasn't been updated yet
        #_top1_, _top5_ = accuracy(results_t, all_labels)
        results_t = torch.cat([all_results, torch.ones_like(all_results[:, :1]).float().cuda()*best_t], 1)
        #top1_, top5_ = accuracy(results_t, all_labels)
        # The threshold for new_whale is appended to the end (of the possibly averaged output)
        # So the last id is for 'new_whale'
        # this was previously set in dataset object as CLASSES*2, so modifying that here
        all_labels[all_labels == CLASSES * 2] = CLASSES
        new_b, known_b, top1, top5, top10 = accuracy(results_t, all_labels, topk=(1,5,10), sep=CLASSES)
    # End of with

    # Classification output
    # Output findings for viewing
    print()
    """
    print("Classification accuracy before add 'new_whale' probability:")
    print("_top1_:", _top1_, "; _top5_:", _top5_)
    print("Classification accuracy before modify 'new_whale' ID:")
    print("top1_:", top1_, "; top5_:", top5_)
    """
    print("Classification accuracy:")
    print("best_t:", best_t)
    print("top1:", top1.cpu().numpy()[0], "; top5:", top5.cpu().numpy()[0], "; top10:", top10.cpu().numpy()[0])

    print("Known whales:")
    print("   top1:", known_b[0].cpu().numpy()[0])
    print("   top5:", known_b[1].cpu().numpy()[0])
    print("   top10:", known_b[2].cpu().numpy()[0])
    print("New whales:")
    print("   top1:", new_b[0].cpu().numpy()[0])
    print("   top5:", new_b[1].cpu().numpy()[0])
    print("   top10:", new_b[2].cpu().numpy()[0])


def run_embedding(model, dst_embed, dataloader_embed, features_file, best_t):
    """
    Steps:
    - Extract embedding features for original and mirrored images of training set
    - Use model's returned features and create a function to compare those against
    all the training features and get the one with the smallest distance (cosine
    similarity should not be necessary as they should already be normalized - see
    notes from talking to Chuck)
    - Organize the results from smallest to largest to get top1 and top5 accuracies
    - Try various threshold for deciding whether 'new_whale'
    (- So all that should really change would be 'torch.sigmoid' part for above fcn,
    adding the threshold for my own threshold, and accuracy method for  my own since
    calculating using min instead of max.)

    Later:
    - Threshold can be picked from doing this in training instead of testing and 
    seeing what works best there.

    """
    # Load the embeddings
    infile = "train_features{}.csv".format(features_file)
    #embeddings = torch.from_numpy(np.loadtxt(infile)).float()
    embeddings = torch.Tensor(pd.read_csv(infile).to_numpy()).float()
    #embeddings = getIDCenters(embeddings)
    # Get mapping of all labels to IDS for embedding later
    #label_id = dst_embed.labels_dict
    infile2 = "train_ids{}.csv".format(features_file)
    #ind_train_toID = torch.from_numpy(np.loadtxt(infile2)).cuda().long()
    ind_train_toID = torch.Tensor(pd.read_csv(infile2).to_numpy()).cuda().long()
    
    # Setup torch and model for evaluation and efficiency
    with torch.no_grad():
        model.eval()
        # Embedded
        all_results_e = []
        all_labels = []
        
        #print("Starting")
        # Run dataset through model and get accuracy
        for data in dataloader_embed:
            #print("In")
            images, labels, _ = data
            images = images.cuda()
            labels = labels.cuda().long()
            # TODO: (Remove if fine) Just removed outs here and names above. shouldn't matter..
            features, _, _ = model(images)

            # Get distances to embeddings
            # dist_mat: pytorch Variable, with shape [N, M]
            dist_mat = euclidean_dist(features.cuda(), embeddings.cuda())
            
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
              dist_mat = (dist_mat[::2, ::2] + dist_mat[1::2, 1::2])/2

            # Currently, what we have is a mapping of distances to images' index but we want
            # a mapping of distances to the ID that corresponds to those images.
            # Instead of creating a new accuracy fcn, if I change the range of the 
            # distance to [0..1], and then do 1 - distances, then I can just use the rest of
            # the set up just like if it were classification. Accuracy fcn was just
            # modified to convert the indices to IDs
            all_results_e.append(dist_mat)
            all_labels.append(labels)
        all_results_e = torch.cat(all_results_e, 0)
        all_labels = torch.cat(all_labels, 0)
        #print("Some results visualized:")
        #see, _ = all_results_e.topk(5, 1, True, True)
        #print(see[:20])
        
        print("Min and Max distances seen: {} {}".format(all_results_e.min(), all_results_e.max()))
        
        min_dist = 0.8
        min_found = all_results_e.min()
        if min_found < min_dist:
            print("#"*20)
            print("Lower min found: {}".format(min_found))
            min_dist = min_found
        results3 = all_results_e - min_dist
        # Don't necessarily need this second part if max_dist is close to 1
        max_dist = 0.9    # 1.7 - 0.8 OR # 0.6 + 0.3
        max_found = results3.max()
        if max_found > max_dist:
            print("#"*20)
            print("Higher max found: {}".format(max_found))
            max_dist = max_found
        results2 = results3 / max_dist
        all_results_e = 1 - results2
    # End of with

    # TODO: implement giving each 'new_whale' their own ID so accuracy metric is more
    # accurate.. (as done on paper w/ explanation to Hilal) if min occ. to include is 1.
    # This won't relly be helpful with 'new_whale's as I can't test for re-identification
    # but I will def need this to include 'new_whale's with more than 1 occurence 
    # to test that the embedding approach works
    
    # There should be no more 'new_whale' in the saved embeddings of the training set
    # as all data in the training set should eventually be considered as an identified whale
    assert ind_train_toID.ne(CLASSES*2).all()
    
    # Also add the mapping for the 'label_map', 
    # as 'new_whale' threshold added to results_t is not
    # a part of the training set data
    label_map = torch.cat([ind_train_toID.view(-1), torch.Tensor([CLASSES]).cuda().long()], 0)
    
    results_t = torch.cat([all_results_e, torch.ones_like(all_results_e[:, :1]).float().cuda()*best_t], 1)
    # The threshold for new_whale is appended to the end (of the possibly averaged output)
    # So the last id is for 'new_whale'
    # this was previously set in dataset object as CLASSES*2, so modifying that here
    all_labels[all_labels == CLASSES * 2] = CLASSES
    new_b, known_b, top1, top5, top10 = accuracy(results_t, all_labels, topk=(1,5,10), label_map=label_map, sep=CLASSES)
    #top1, top5 = accuracy(results_t, all_labels)
    
    # Embedding output
    print()
    print("Embedding accuracy:")
    print("best_t:", best_t)
    print("top1:", top1.cpu().numpy()[0], "; top5:", top5.cpu().numpy()[0], "; top10:", top10.cpu().numpy()[0])

    print("Known whales:")
    print("   top1:", known_b[0].cpu().numpy()[0])
    print("   top5:", known_b[1].cpu().numpy()[0])
    print("   top10:", known_b[2].cpu().numpy()[0])
    print("New whales:")
    print("   top1:", new_b[0].cpu().numpy()[0])
    print("   top5:", new_b[1].cpu().numpy()[0])
    print("   top10:", new_b[2].cpu().numpy()[0])
    

def test_embed(checkPoint_start=0, fold="newWhale3_less", model_name='senet154', flip_lr=False, embed=False, features_file="", testing=True):
    # Get the model
    device = torch.device('cuda')
    model = model_whale(num_classes=CLASSES * 2, inchannels=4, model_name=model_name).to(device)
    # Find result dir
    resultDir = './result/{}_{}'.format(model_name, fold_index)
    checkPoint = os.path.join(resultDir, 'checkpoint')
    # Don't want to keep track of the predictions for each image for now
    #npy_dir = resultDir + '/out_{}'.format(checkPoint_start)
    # Create output dirs
    #os.makedirs(npy_dir, exist_ok=True)
    
    # Load the pretrained weights
    if not checkPoint_start == 0:
        #model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start)),skip=[])
        ckp = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (checkPoint_start)))
        model.load_state_dict(torch.load(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start))))
        #model.load_state_dict(ckp['optimizer'])
        best_t = ckp['best_t']
        #print('\nbest_t:', best_t)
   
    # Setup dataloader
    # Normal classifier or embedding tests
    if testing:
        data_test = pd.read_csv('./input/test_split_{}.csv'.format(fold_index))
        names_embed = data_test['Image'].tolist()
        labels_embed = data_test['Id'].tolist()
        batch_size = 16
        mode = 'embed'
    # Final embedding tests
    else:
        data_embed = pd.read_csv('./input/embed_split_{}_test.csv'.format(fold_index))
        names_embed = data_embed['Image'].tolist()
        labels_embed = data_embed['Id'].tolist()
        batch_size = 16
        mode = 'embed'
    print("\nNumber of images to test:", len(names_embed))
    
    # If including flipped images, process data a little differently
    # Use different transform and collate_fn
    if flip_lr:
        dst_embed = WhaleTestDataset(names_embed, labels_embed, mode=mode, transform=transform)
        dataloader_embed = DataLoader(dst_embed, batch_size=batch_size, num_workers=8, collate_fn=embed_collate)
    else:
        dst_embed = WhaleTestDataset(names_embed, labels_embed, mode=mode, transform=transform_embed)
        dataloader_embed = DataLoader(dst_embed, batch_size=batch_size, num_workers=8)
   
    # Run the appropriate classifier or embedding
    if embed:
        run_embedding(model, dst_embed, dataloader_embed, features_file, best_t)
    else:
        run_classifier(model, dataloader_embed, best_t)
    

if __name__ == '__main__':
    # Initial code
    """
    checkpoint_start = 0
    fold_index = 1
    test(checkPoint_start, fold_index, model_name)
    """

    # Relevant vars
    flip_lr = True
    model_name = 'senet154'
    # 9999 Changed this and various others below
    fold_index = "final2" 
    #fold_index = "new_flukes" 
    features_file = ""
    # Normal testing option where method is nearest neighbors in embedding space
    embed = True
    # True if normal testing, otherwise doing final embedding tests
    testing = False 
    
    # Vars for test outputs from earlier experiments
    #checkPoint_start = 6000
    
    # Vars for final experiments
    #"""
    if embed:
        checkPoint_start = 8600
        features_file = "_final2"
    else:
        checkPoint_start = 8600
    if not(testing):
        embed = True
        checkPoint_start = 8600
        features_file = "_final2_added"
    #"""

    print("Model name:", model_name)
    print("Fold:", fold_index)
    print("Checkpoint start:", checkPoint_start)
    print("Also tests flipped images:", flip_lr)
    print("Embedding?", embed)
    print("Getting features (if embed) from files with ending:", features_file)
    print("Doing final embedding tests?", not(testing))
    test_embed(checkPoint_start, fold_index, model_name, flip_lr, embed, features_file, testing)

