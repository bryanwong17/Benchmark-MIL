import torch.nn as nn

from utils import switch_dim, get_metrics
from pl_model.mil_trainer import MILTrainerModule

# def apply_sparse_init(m):
#     if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
#         nn.init.orthogonal_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)

def get_mil_model(mil_model, num_feats, num_classes, loss_weight=None):
    if mil_model in ["mean_pooling", "max_pooling", "ABMIL", "GABMIL"]:
        if mil_model == "mean_pooling":
            pooling_layer = nn.AdaptiveAvgPool1d(1)
        elif mil_model == "max_pooling":
            pooling_layer = nn.AdaptiveMaxPool1d(1)
        elif mil_model == "ABMIL":
            from model.abmil import AttentionPooling
            pooling_layer = AttentionPooling(num_feats, mid_dim=128, out_dim=1, flatten=True, dropout=0.)
        elif mil_model == "GABMIL":
            from model.gabmil import GatedAttentionPooling
            pooling_layer = GatedAttentionPooling(num_feats, mid_dim=128, out_dim=1, flatten=True, dropout=0.)
        
        classifier_model = nn.Sequential(
            switch_dim(),
            pooling_layer,
            nn.Flatten(),
            nn.Linear(num_feats, num_classes)
        )

        loss = nn.CrossEntropyLoss(weight=loss_weight)

    elif mil_model == "DSMIL":
        from model.dsmil import FCLayer, BClassifier, MILNet
        i_classifier = FCLayer(in_size=num_feats, out_size=num_classes) # consider all classes as positive (out_size=num_classes)
        b_classifier = BClassifier(input_size=num_feats, output_class=num_classes, dropout_v=0.)
        classifier_model = MILNet(i_classifier, b_classifier)
        # classifier_model.apply(lambda m: apply_sparse_init(m))

        loss = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
    
    elif mil_model in ["CLAM-SB", "CLAM-MB"]:
        from model.clam import CLAM_SB, CLAM_MB
        CLAM = CLAM_SB if mil_model == "CLAM-SB" else CLAM_MB
        clam_model_dict = {"dropout": True, "n_classes": num_classes, "subtyping": True, "size_arg": "small",
                           "k_sample": 8, "bag_weight": 0.7, "embed_dim": num_feats}
        classifier_model = CLAM(**clam_model_dict, instance_loss_fn="svm")
        
        loss = nn.CrossEntropyLoss(weight=loss_weight)
    
    elif mil_model == "TransMIL":
        from model.transmil import TransMIL
        classifier_model = TransMIL(n_classes=num_classes, input_size=num_feats)

        loss = nn.CrossEntropyLoss(weight=loss_weight)
    
    elif mil_model == "DTFD-MIL":
        from model.dtfdmil.network import DimReduction
        from model.dtfdmil.attention import Attention_Gated as Attention
        from model.dtfdmil.attention import Attention_with_Classifier, Classifier_1fc

        mDim = num_feats // 2

        DTFDclassifier = Classifier_1fc(mDim, num_classes, 0.0)
        DTFDattention = Attention(mDim)
        DTFDdimReduction = DimReduction(num_feats, mDim, numLayer_Res=0)
        DTFDattCls = Attention_with_Classifier(L=mDim, num_cls=num_classes, droprate=0.0)
        classifier_model = [DTFDclassifier, DTFDattention, DTFDdimReduction, DTFDattCls]

        loss0 = nn.CrossEntropyLoss(reduction="none", weight=loss_weight)
        loss1 = nn.CrossEntropyLoss(reduction="none", weight=loss_weight)
        loss = [loss0, loss1]

    return classifier_model, loss

def get_model_module(args, seed, class_names_list, mil_model, num_feats, num_classes, loss_weight=None):
    from pl_model.forward_fn import get_forward_func
    task = "multiclass"
    classifier_model, loss = get_mil_model(mil_model, num_feats, num_classes, loss_weight=loss_weight)
    forward_func = get_forward_func(mil_model)

    if mil_model == "DTFD-MIL":
        from pl_model.mil_trainer_dtfdmil import DTFDTrainerModule
        trainer_module = DTFDTrainerModule(args, seed, class_names_list, classifier_model, loss, get_metrics(num_classes, task))
    else:
        trainer_module = MILTrainerModule(args, seed, class_names_list, classifier_model, loss, get_metrics(num_classes, task), num_classes, forward_func=forward_func)
                                    
    return trainer_module