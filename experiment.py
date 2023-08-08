import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import cv2
from operator import itemgetter


# get sub-modules
from trainer.config import(
    SystemConfig,
    DatasetConfig,
    DataloaderConfig,
    OptimizerConfig,
    TrainerConfig   
)

from trainer.utils import(
    setup_system,
    get_camvid_dataset_parameters,
    draw_semantic_segmentation_batch,
    patch_configs
)
from trainer.loss import SemanticSegmentationLoss
from trainer.model import SemSeg
from trainer.eval_metrics import IntersectionOverUnion
from trainer.dataset import SemSegDataset
from torch.utils.data import DataLoader
from trainer.trainer import Trainer
from trainer.plots import (
    plot_iou_eachclass,
    plot_loss_and_metric_curves
)
#augmentations

from albumentations import Compose,HorizontalFlip,ShiftScaleRotate,HueSaturationValue,Normalize,RandomCrop
from albumentations.pytorch import ToTensorV2

class Experiment:
    def __init__(
            self,
            system_config: SystemConfig = SystemConfig(),
            dataset_config : DatasetConfig = DatasetConfig(),
            dataloader_config: DataloaderConfig = DataloaderConfig(),
            optimizer_config : OptimizerConfig = OptimizerConfig()
    ):
        self.system_config = system_config

        # train_dataset
        train_dataset = SemSegDataset(**get_camvid_dataset_parameters(
                                                dataset_config.root_dir,
                                                dataset_type="train",
                                                transforms=Compose([
                                                    HorizontalFlip(),
                                                    ShiftScaleRotate(
                                                        shift_limit=0.0625,
                                                        scale_limit=0.1,
                                                        rotate_limit=45,
                                                        border_mode=cv2.BORDER_CONSTANT,
                                                        value=0,
                                                        mask_value=11,
                                                        p=0.75
                                                    ),
                                                    HueSaturationValue(),
                                                    RandomCrop(height=352,width=480),
                                                    Normalize(),
                                                    ToTensorV2()
                                                ])                                                
                                        )
                        )
        
        self.train_loader = DataLoader(train_dataset,
                        dataloader_config.batch_size,
                        shuffle=True,
                        num_workers= dataloader_config.num_workers,
                        pin_memory=True
                        )
        
        test_dataset =  SemSegDataset(
                         **get_camvid_dataset_parameters(
                                data_path=dataset_config.root_dir,
                                dataset_type="test",
                                transforms=Compose([
                                    Normalize(),
                                    ToTensorV2()
                                ])
                        
                            )
                        )
        
        self.test_loader = DataLoader(test_dataset,
                   batch_size=dataloader_config.batch_size,
                   shuffle=False,
                   num_workers=dataloader_config.num_workers,
                   pin_memory=True
                   )
        
        self.num_classes = self.test_loader.dataset.get_num_classes()
                        
        # get model
        self.model = SemSeg(self.num_classes, final_upsample=True)

        # get loss_fn
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.num_classes)
        # get metric_fn
        self.metric_fn = IntersectionOverUnion(self.num_classes,
                                               reduced_probs=False
        )
        # get_optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr = optimizer_config.learning_rate,
            weight_decay= optimizer_config.weight_decay,
        )

        self.lr_scheduler = MultiStepLR(
            self.optimizer,
            milestones= optimizer_config.lr_step_milestones,
            gamma=optimizer_config.lr_gamma
        )

        # self.visualizer = TensorBoardVisu


    def run(self,trainer_config = TrainerConfig):
        setup_system(self.system_config)
        
        device = trainer_config.device

        print(device)

        # send model and loss fn to device

        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)

        trainer = Trainer(
            model=self.model,
            train_loader= self.train_loader,
            test_loader=self.test_loader,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            lr_scheduler= self.lr_scheduler,
            data_getter= itemgetter('image'),
            target_getter=itemgetter("mask"),
            device= device,
            get_key_metric= itemgetter("miou"),
        )

        self.metrics = trainer.fit(trainer_config.num_epochs)

        
        return self.metrics




# class ExperimentWithSemanticSegmentationLoss(Experiment):
#     def __init__(self, *args, **kwargs):
#         # init fields
#         super().__init__(*args, **kwargs)
#         num_classes = self.test_loader.dataset.get_num_classes()
#         self.loss_fn = SemanticSegmentationLoss(num_classes=num_classes, ignore_indices=num_classes)






# start experiment
dataloader_config, trainer_config = patch_configs(epoch_num_to_set=30, batch_size_to_set=16, num_workers_to_set=0)

optimizer_config = OptimizerConfig(learning_rate=1e-3, lr_step_milestones =[], weight_decay=4e-5)



experiment = Experiment(dataloader_config=dataloader_config, optimizer_config=optimizer_config)
metrics = experiment.run(trainer_config)


# semseg_loss_experiment = ExperimentWithSemanticSegmentationLoss(dataloader_config=dataloader_config)
# metrics = semseg_loss_experiment.run(trainer_config)

torch.save(metrics, "metrics/metrics_dict.pth")

plot_iou_eachclass(11,metrics)
plot_loss_and_metric_curves(metrics)
