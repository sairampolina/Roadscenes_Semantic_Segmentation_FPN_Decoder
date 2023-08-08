
import torch
import os
import datetime

from typing import Union,Callable
from operator import itemgetter
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from .hooks import train_hook_default,test_hook_default

class Trainer:

    def __init__(
            self,
            model : torch.nn.Module,
            train_loader : torch.utils.data.DataLoader,
            test_loader : torch.utils.data.DataLoader,
            loss_fn : Callable,
            metric_fn : Callable,
            optimizer : torch.optim.Optimizer,
            lr_scheduler : torch.optim.lr_scheduler,
            data_getter: Callable = itemgetter("image"),
            target_getter : Callable = itemgetter("target"),
            device: Union[torch.device, str] = "cuda",
            get_key_metric:Callable = itemgetter("miou"),
            model_save_best: bool = True,
            model_saving_frequency: int = 1,
            save_dir: Union[str,Path] = "checkpoints",
            stage_progress: bool = True,
            visualizer = None
    ):
        
        self.model = model

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.loss_fn = loss_fn
        self.metric_fn = metric_fn

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.data_getter = data_getter
        self.target_getter = target_getter
        self.device = device

        self.model_save_best = model_save_best
        self.model_saving_frequency = model_saving_frequency
        self.save_dir =save_dir

        self.stage_progress = stage_progress
        self.visualizer = visualizer

        self.hooks = {}
        self.metrics = {"epoch": [],"train_loss": [],"test_loss" : [], "test_metric":[]}
        self.get_key_metric = get_key_metric
        self._register_default_hooks()

        if model_save_best or model_saving_frequency:
            os.makedirs(self.save_dir,exist_ok=True)


    def fit(self, epochs):

        epoch_iter = tqdm(range(epochs),dynamic_ncols=True)

        best_metric = 0

        for epoch in epoch_iter:
            # go through train_loader and find avg_loss
            output_train = self.hooks["train"](
                        self.model,
                        self.train_loader,
                        self.loss_fn,
                        self.optimizer,
                        self.data_getter,
                        self.target_getter,
                        self.device,
                        self.stage_progress,
                        prefix = f"[{epoch}/{epochs}]"
                    )
            
            # go through test_loader and find avg_loss and metric_value

            output_test = self.hooks["test"](
                        self.model,
                        self.test_loader,
                        self.loss_fn,
                        self.metric_fn,
                        self.optimizer,
                        self.data_getter,
                        self.target_getter,
                        self.device,
                        self.stage_progress,
                        self.get_key_metric,
                        prefix = f"[{epoch}/{epochs}]"    
                    )
            
            # add metrics
            self.metrics["epoch"].append(epoch)
            self.metrics["train_loss"].append(output_train["loss"])
            self.metrics["test_metric"].append(output_test["metric"])
            self.metrics["test_loss"].append(output_test["loss"])

            # modify LR
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(output_train["loss"])
            else:
                self.lr_scheduler.step()

            # model-saving
            if self.model_save_best:
                cur_metric = self.get_key_metric(output_test["metric"])

                if cur_metric > best_metric:
                    torch.save(self.model.state_dict(),
                               os.path.join(self.save_dir,self.__class__.__name__)+'_best.pth')
                    best_metric = cur_metric
                    
            elif (epoch+1)/ self.model_saving_frequency == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_dir, self.model.__class__.__name__) + '_' +
                            str(datetime.datetime.now()) + '.pth')
    
        return self.metrics


    def _register_default_hooks(self):
        self.hooks["train"] = train_hook_default
        self.hooks["test"] = test_hook_default
        self.hooks["end_epoch"] = None

