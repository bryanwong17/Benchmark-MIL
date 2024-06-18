import torch
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_optimizer import Lookahead

class MILTrainerModule(pl.LightningModule):
    def __init__(self, args, classifier, loss, metrics, num_classes, forward_func="general"):
        super(MILTrainerModule, self).__init__()

        self.args = args
        
        # Lookahead optimizer (TransMIL) can only work with automatic optimization
        if self.args.mil_model != "TransMIL":
            self.automatic_optimization = False # default automatic optimization is True

        self.classifier = classifier
        self.loss = loss
        self.metrics = metrics

        self.num_classes = num_classes
        
        self.forward_func = forward_func

        self.accumulate_grad_batches = args.accumulate_grad_batches

        self.train_metrics = metrics.clone(postfix='/train')
        self.val_metrics = nn.ModuleList([metrics.clone(postfix='/val'), metrics.clone(postfix='/test')])
        self.test_metrics = metrics.clone(prefix='final_test/')
        
        self.save_hyperparameters("args")
    
    def classifier_forward(self, data, caption=None, label=None):
        return self.forward_func(data, self.classifier, self.loss, self.num_classes, caption=caption, label=label)
    
    def forward(self, feats, caption, label=None, train=False):
        
        bag_prediction, loss, Y_prob = self.classifier_forward(feats, caption, label) # depends on the mil aggregator type
      
        if train and self.args.mil_model != "TransMIL":
            self.manual_backward(loss) # manual backward classifier

        return bag_prediction, loss, Y_prob
    
    def training_step(self, batch, batch_idx):
        feats, caption, label = batch
        feats = feats.squeeze(0) # remove the batch size (1)
      
        y, loss, y_prob = self.forward(feats, caption, label, train=True)

        opt = self.optimizers()
        
        if self.args.mil_model != "TransMIL":

           if (batch_idx + 1) % self.accumulate_grad_batches == 0:
               opt.step()
               opt.zero_grad()
        
        self.log("Loss/train", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.args.batch_size)

        # https://github.com/Lightning-AI/pytorch-lightning/issues/2210
        self.train_metrics.update(y_prob, label) # metrics are calculated per epoch, not per step (batch) -> solution: use update
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.args.batch_size)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        feats, caption, label = batch
        feats = feats.squeeze(0) # remove the batch size (1)
     
        y, loss, y_prob = self.forward(feats, caption, label, train=False)

        if not self.trainer.sanity_checking:
            prefix = get_prefix_from_val_id(dataloader_idx) # val/test
            metrics_idx = dataloader_idx if dataloader_idx is not None else 0
            self.log("Loss/%s" % prefix, loss, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False, batch_size=self.args.batch_size)
            self.val_metrics[metrics_idx].update(y_prob, label)
            self.log_dict(self.val_metrics[metrics_idx], on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False, batch_size=self.args.batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        feats, caption, label = batch
        feats = feats.squeeze(0) # remove the batch size (1)
       
        y, loss, y_prob = self.forward(feats, caption, label, train=False)

        self.log("Loss/final_test", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.args.batch_size)
        self.test_metrics.update(y_prob, label)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.args.batch_size)
        return self.test_metrics

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None and self.args.mil_model != "TransMIL":
            sch.step()
    
    def configure_optimizers(self):
        params = [{"params": filter(lambda p: p.requires_grad, self.classifier.parameters())}]
        
        if self.args.opt == "adam":
            cus_optimizer = torch.optim.Adam(params, lr=self.args.lr, betas=(0.5, 0.9), weight_decay=self.args.weight_decay)
        elif self.args.opt == "adamw":
            cus_optimizer = torch.optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.opt == "lookahead_radam" and self.args.mil_model == "TransMIL":
            base_optimizer = torch.optim.RAdam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
            cus_optimizer = Lookahead(base_optimizer)

        cus_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cus_optimizer, T_max=self.args.epochs, eta_min=5e-6)

        return {
            "optimizer": cus_optimizer,
            "lr_scheduler": cus_scheduler,
        }
    
def get_prefix_from_val_id(dataloader_idx):
    if dataloader_idx is None or dataloader_idx == 0:
        return "val"
    elif dataloader_idx == 1:
        return "test"
    else:
        return NotImplementedError
