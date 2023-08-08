from abc import ABC,abstractmethod
import numpy as np

import torch

from typing import Iterable,Union

class BaseMetric(ABC):

    @abstractmethod
    def update_values(self, preds, targets):
        pass
    
    @abstractmethod
    def get_metric_value(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass



class ConfusionMatrix(BaseMetric):

    def __init__(self, num_classes : int,normalized:bool = False):

        self.num_classes = num_classes
        self.normalized = normalized
        self.conf_matrix = np.ndarray(shape=(self.num_classes, self.num_classes),dtype=np.uint16)
        self.reset()
    
    def reset(self):
        
        self.conf_matrix.fill(0)
    
    def update_values(self, preds, targets):
        """
        args:
        preds(torch.Tensor) : (B, num_classes, H,W)
        targets(torch.Tensor) : (B, H, W)
        """

        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()

        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()

        valid_ids = (targets >=0) & (targets < self.num_classes)
        
        targets = targets[valid_ids]
        preds = preds[valid_ids]
        
        replace_mat = np.hstack((targets.flatten().reshape(-1,1), preds.flatten().reshape(-1,1)))
        
        conf,_ = np.histogramdd(replace_mat,
                       bins = (self.num_classes,self.num_classes),
                       range=[(0,self.num_classes),(0,self.num_classes)])

        self.conf_matrix = self.conf_matrix + conf.astype(np.uint16)
    
    def get_metric_value(self):

        if self.normalized:
            conf = self.conf_matrix.astype(np.float16)
            return  conf / conf.sum(1).min(1e-12)[:,None]

        return self.conf_matrix
    

class IntersectionOverUnion(BaseMetric):

    def __init__(
            self, num_classes: int, 
            normalized_flag : bool = False,
            reduced_probs: bool = False,
            ignore_indices : Union[Iterable[int], int, None] = None ) -> 'dict[str,float]':
        
        self.conf_matrix = ConfusionMatrix(num_classes,normalized= normalized_flag)

        self.reduced_probs = reduced_probs

        self.ignore_indices = ignore_indices


    def reset(self):
        self.conf_matrix.reset()

    def update_values(self, preds : torch.Tensor, targets : torch.Tensor):
        
        if not self.reduced_probs:
            preds = preds.argmax(dim=1)

        self.conf_matrix.update_values(preds, targets)


    def get_metric_value(self):
        conf_mat = self.conf_matrix.get_metric_value()

        if self.ignore_indices is not None:

            conf_mat[:, self.ignore_indices] = 0
            conf_mat[self.ignore_indices, :] = 0


        TP = np.diag(conf_mat)
        FN = conf_mat.sum(axis = 1) - TP
        FP = conf_mat.sum(axis = 0) - TP

        with np.errstate(divide='ignore', invalid='ignore'):
            iou = TP/(TP + FP + FN)

        if self.ignore_indices is not None:
            iou_classes = np.delete(iou, self.ignore_indices)
            miou = np.nanmean(iou_classes)
            
        else:
            miou = np.nanmean(iou)
        return {"miou": miou, "iou" : iou}