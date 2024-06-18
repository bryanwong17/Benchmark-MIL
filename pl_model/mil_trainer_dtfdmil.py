import torch
import torch.nn as nn
import pytorch_lightning as pl

from optimizer.radam import RAdam
from optimizer.lookahead import Lookahead

from pl_model.forward_fn import dtfdmil_forward_1st_tier, dtfdmil_forward_2nd_tier


class DTFDTrainerModule(pl.LightningModule):
    def __init__(self, args, classifier_list, loss_list, metrics):
        super(DTFDTrainerModule, self).__init__()

        self.args = args

        self.automatic_optimization = False

        self.classifier = classifier_list[0]
        self.attention = classifier_list[1]
        self.dimReduction = classifier_list[2]
        self.UClassifier = classifier_list[3]

        self.loss0 = loss_list[0]
        self.loss1 = loss_list[1]

        self.metrics = metrics

        self.accumulate_grad_batches = args.accumulate_grad_batches

        self.train_metrics = metrics.clone(postfix='/train')
        self.val_metrics = nn.ModuleList([metrics.clone(postfix='/val'), metrics.clone(postfix='/test')])
        self.test_metrics = metrics.clone(prefix='final_test/')
        
        self.save_hyperparameters("args")
    
    def forward(self, feats, caption=None, label=None, train=False):

        loss0, slide_pseudo_feat = dtfdmil_forward_1st_tier(self.args, feats, self.classifier, self.attention, 
                                                                    self.dimReduction, self.loss0, caption, label)
        
        if train:
            self.manual_backward(loss0, retain_graph=True) # retain_graph=True -> used when want to backward through the graph a second time
            torch.nn.utils.clip_grad_norm_(self.dimReduction.parameters(), self.args.grad_clipping)
            torch.nn.utils.clip_grad_norm_(self.attention.parameters(), self.args.grad_clipping)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.args.grad_clipping)

        loss1, gSlidePred = dtfdmil_forward_2nd_tier(slide_pseudo_feat, self.UClassifier, self.loss1, caption, label)

        if train:
            self.manual_backward(loss1)
            torch.nn.utils.clip_grad_norm_(self.UClassifier.parameters(), self.args.grad_clipping)

        return loss0, loss1, gSlidePred
    
    def training_step(self, batch, batch_idx):
        feats, caption, label = batch
        feats = feats.squeeze(0) # remove the batch size (1)

        loss0, loss1, Y_prob = self.forward(feats, caption, label, train=True) # Y_prob not being softmax?

        optimizer0, optimizer1 = self.optimizers()

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            optimizer0.step()
            optimizer1.step()
            optimizer0.zero_grad()
            optimizer1.zero_grad()
        
        total_loss = loss0 + loss1
        self.log("Loss/train", total_loss, on_step=True, on_epoch=True, sync_dist=True)

        # https://github.com/Lightning-AI/pytorch-lightning/issues/2210
        self.train_metrics.update(Y_prob, label) # metrics are calculated per epoch, not per step (batch) -> solution: use update
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        feats, caption, label = batch
        feats = feats.squeeze(0) # remove the batch size (1)
     
        loss0, loss1, Y_prob = self.forward(feats, caption, label, train=False)

        if not self.trainer.sanity_checking:
            prefix = get_prefix_from_val_id(dataloader_idx) # val/test
            metrics_idx = dataloader_idx if dataloader_idx is not None else 0
            total_loss = loss0 + loss1
            self.log("Loss/%s" % prefix, total_loss, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            self.val_metrics[metrics_idx].update(Y_prob, label)
            self.log_dict(self.val_metrics[metrics_idx], on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)

        return total_loss

    def test_step(self, batch, batch_idx):
        feats, caption, label = batch
        feats = feats.squeeze(0) # remove the batch size (1)
       
        loss0, loss1, Y_prob = self.forward(feats, caption, label, train=False)

        total_loss = loss0 + loss1
        self.log("Loss/final_test", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics.update(Y_prob, label)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, sync_dist=True)
        return self.test_metrics

    def on_train_epoch_end(self):
        scheduler0, scheduler1 = self.lr_schedulers()
        if scheduler0 is not None and scheduler1 is not None:
            scheduler0.step()
            scheduler1.step()
    
    def configure_optimizers(self):
        trainable_parameters = []
        trainable_parameters += list(self.classifier.parameters())
        trainable_parameters += list(self.attention.parameters())
        trainable_parameters += list(self.dimReduction.parameters())

        params_optimizer_0 = [{"params": filter(lambda p: p.requires_grad, trainable_parameters)}]
        params_optimizer_1 = [{"params": filter(lambda p: p.requires_grad, self.UClassifier.parameters())}]

        optimizer0 = torch.optim.Adam(params_optimizer_0, lr=self.args.lr, weight_decay=self.args.weight_decay)
        optimizer1 = torch.optim.Adam(params_optimizer_1, lr=self.args.lr, weight_decay=self.args.weight_decay)

        scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer0, [int(self.args.epochs/2)], gamma=self.args.lr_decay_ratio)
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [int(self.args.epochs/2)], gamma=self.args.lr_decay_ratio)

        return [
            {"optimizer": optimizer0, "lr_scheduler": scheduler0},
            {"optimizer": optimizer1, "lr_scheduler": scheduler1}
        ]

def get_prefix_from_val_id(dataloader_idx):
    if dataloader_idx is None or dataloader_idx == 0:
        return "val"
    elif dataloader_idx == 1:
        return "test"
    else:
        return NotImplementedError
