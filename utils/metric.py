import numpy as np
import pandas as pd
import torch
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
def accuracy(output, target, topk=(1, 5), label_map=None, sep=None, mode=None):
    """
    Computes the precision@k for the specified values of k
    This ignores the number of guesses.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    probs, pred = output.topk(maxk, 1, True, True)
    #_, pred = output.topk(maxk, 1, True, True)
    #print("Probabilities. Trying to see") 
    # TODO: Look at this
    # if top ones are exactly 1 since should be the same vector
    # if yes, then might be easier to just ignore the first result, if perfect, on train...
    #print(probs)
    
    # If this comes from embedding, we need to get the correct labels first
    # Convert the labels from the index of the train image to its ID for the model
    if (type(label_map) != type(None)):
        #print()
        """
        print("Original location of label")
        print(pred)
        print("Location to label")
        print(label_map)
        """
        pred = label_map[pred.long()]
    """
    print("Gotten prediction")
    print(pred)
    #exit()
    print("Correct prediction")
    print(target)
    """

    res = []
    # If it is requested to split the accuracy into known vs new_whale
    if sep:
        new_whales = target.eq(sep)
        known = new_whales.eq(0)
        new_target = target[new_whales]
        new_pred = pred[new_whales]
        known_target = target[known]
        known_pred = pred[known]
        
        """
        to_see = probs[new_whales][:10]
        print("Helpful info to figure out appropriate threshold")
        print("New whales")
        print(to_see)
        #""
        print("Known whales")
        to_see = probs[known][:10]
        print(to_see)
        """
        #print("No. of new_whales and known whales: {} {}".format(new_target.size(0), known_target.size(0)))
        #exit()
        
        # Get the new_whale accuracy
        new_pred = new_pred.t()
        new_correct = new_pred.eq(new_target.view(1, -1).expand_as(new_pred))

        # Calculate the actual percentage accuracy value - i.e. precision
        batch_size2 = new_target.size(0)
        res2 = []
        if batch_size2 > 0:
            for k in topk:
                correct_k = new_correct[:k].any(0).view(-1).float().sum(0, keepdim=True)
                res2.append(correct_k.mul_(100.0 / batch_size2))
        else:
            for k in topk:
                res2.append(torch.Tensor([0]).cuda())
        res.append(res2)
    
        # Get the known whale accuracy
        known_pred = known_pred.t()
        known_correct = known_pred.eq(known_target.view(1, -1).expand_as(known_pred))

        # Calculate the actual percentage accuracy value - i.e. precision
        batch_size2 = known_target.size(0)
        res2 = []
        if batch_size2 > 0:
            for k in topk:
                correct_k = known_correct[:k].any(0).view(-1).float().sum(0, keepdim=True)
                res2.append(correct_k.mul_(100.0 / batch_size2))
        else:
            for k in topk:
                res2.append(torch.Tensor([0]).cuda())
        res.append(res2)

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # Calculate the actual percentage accuracy value - i.e. precision
    for k in topk:
        # In case label can be repeated in the preddictions (in the case of embedding)
        # Make sure to only count the correct answer once
        correct_k = correct[:k].any(0).view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_top_k(output, topk=(3,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    return pred

def apk(actual, predicted, k=10):
    """ 
    Computes average precision at k: 
    Sees if any of top k predictions matches the
    1 actual, but takes into account number of
    guesses before reaching actual correct one..
    https://rdrr.io/cran/Metrics/man/apk.html
    """
    actual = [int(actual)]
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        print("Somehow not actual")
        return 0.0
    
    # print(actual)
    return score / min(len(actual), k)

def mapk(actual, predicted, k=10, label_map=None):
    """ Computes the mean average precision. """
    _, predicted = predicted.topk(k, 1, True, True)
    # If this comes from embedding, we need to get the correct labels first
    # Convert the labels from the index of the train image to its ID for the model
    if (type(label_map) != type(None)):
        predicted = label_map[predicted.long()]
    actual = actual.data.cpu().numpy()
    predicted = predicted.data.cpu().numpy()
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
