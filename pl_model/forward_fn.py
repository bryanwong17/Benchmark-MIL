import random
import numpy as np

import torch
import torch.nn.functional as F

from utils import get_cam_1d

def get_forward_func(mil_model):
    if mil_model in ["meanpooling", "maxpooling", "ABMIL", "GABMIL"]:
        return general_forward
    elif mil_model == "DSMIL":
        return dsmil_forward
    elif mil_model in ["CLAM-SB", "CLAM-MB"]:
        return clam_forward
    elif mil_model == "TransMIL":
        return transmil_forward
    elif mil_model == "DTFD-MIL":
        pass
    else:
        raise NotImplementedError

def general_forward(data, classifier, loss, num_classes, caption=None, label=None):
    pred = classifier(data)
    if label is None:
        return pred
    else:
        loss = loss(pred, label)
        pred_prob = F.softmax(pred, dim=1)
        return pred, loss, pred_prob

def dsmil_forward(data, classifier, loss, num_classes, caption=None, label=None):
    with torch.no_grad():
        label = F.one_hot(label, num_classes).float() # consider all classes as positive

    ins_prediction, bag_prediction, _, _ = classifier(data)
    max_prediction, _ = torch.max(ins_prediction, 0)
    bag_loss = loss(bag_prediction.view(1, -1), label.view(1, -1))
    max_loss = loss(max_prediction.view(1, -1), label.view(1, -1))
    loss = 0.5 * bag_loss + 0.5 * max_loss

    Y_prob = torch.sigmoid(bag_prediction)
    return bag_prediction, loss, Y_prob

def clam_forward(data, classifier, loss, num_classes, caption=None, label=None):
    logits, Y_prob, _, _, instance_dict = classifier(data, label=label, instance_eval=True)
    loss = loss(logits, label)
    instance_loss = instance_dict["instance_loss"]
    total_loss = classifier.bag_weight * loss + (1 - classifier.bag_weight) * instance_loss
    return logits, total_loss, Y_prob

def transmil_forward(data, classifier, loss, num_classes, caption=None, label=None):
    logits, Y_prob, _ = classifier(data)
    loss = loss(logits, label)
    return logits, loss, Y_prob

def dtfdmil_forward_1st_tier(args, data, classifier, attention, dimReduction, loss0, caption=None, label=None):
    
    instance_per_group = args.total_instance // args.numGroup

    slide_pseudo_feat = []
    slide_sub_preds = []
    slide_sub_labels = []

    feat_index = list(range(data.shape[0]))
    random.shuffle(feat_index)
    index_chunk_list = np.array_split(np.array(feat_index), args.numGroup)
    index_chunk_list = [sst.tolist() for sst in index_chunk_list]
    for tindex in index_chunk_list:
        slide_sub_labels.append(label)
        subFeat_tensor = torch.index_select(data, dim=0, index=torch.LongTensor(tindex).to(data.device))
        tmidFeat = dimReduction(subFeat_tensor)
        tAA = attention(tmidFeat).squeeze(0)
        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
        
        tPredict = classifier(tattFeat_tensor)  ### 1 x 2
        slide_sub_preds.append(tPredict)

        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1) ## n x cls
        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1) ## n x cls

        _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True) # CHANGED: It will sort the probability depending on the PSEUDO BAG CLASS labels
        sort_idx = sort_idx.flatten() # ADDED
        
        topk_idx_max = sort_idx[:instance_per_group].long()
        topk_idx_min = sort_idx[-instance_per_group:].long()
        topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

        MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
        max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
        af_inst_feat = tattFeat_tensor

        if args.distill == "MaxMinS":
            slide_pseudo_feat.append(MaxMin_inst_feat)
        elif args.distill == "MaxS":
            slide_pseudo_feat.append(max_inst_feat)
        elif args.distill == "AFS":
            slide_pseudo_feat.append(af_inst_feat)
    
    slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0) ### numGroup x fs
    
    slide_sub_preds = torch.cat(slide_sub_preds) ### numGroup x fs
    slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
    loss = loss0(slide_sub_preds, slide_sub_labels).mean()

    return loss, slide_pseudo_feat

def dtfdmil_forward_2nd_tier(slide_pseudo_feat, UClassifier, loss1, caption=None, label=None):
    gSlidePred = UClassifier(slide_pseudo_feat)
    loss = loss1(gSlidePred, label).mean()
    return loss, gSlidePred
