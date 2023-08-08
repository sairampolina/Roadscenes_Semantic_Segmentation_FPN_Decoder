import git
import sys
import torch

import os
import tempfile
import shutil

import numpy as np
import matplotlib.pyplot as plt


from .config import(
    SystemConfig,
    DatasetConfig,
    DataloaderConfig,
    OptimizerConfig,
    TrainerConfig   
)
def download_git_folder(git_url, src_folder, dst_folder):
    """ Download folder from remote git repo

    Arguments:
        git_url (string): url of remote git repository.
        src_folder (string): path to required folder (related of git repo root).
        dst_folder (string): destination path for required folder (local path).
    """
    class Progress(git.RemoteProgress):
        def update(self, op_code, cur_count, max_count=None, message=''):  # pylint: disable=unused-argument
            sys.stdout.write('\r')
            sys.stdout.write('Download: {}'.format(message).ljust(80, ' '))
            sys.stdout.flush()

    tmp = tempfile.mkdtemp()
    git.Repo.clone_from(git_url, tmp, branch='master', depth=1, progress=Progress())
    shutil.move(os.path.join(tmp, src_folder), dst_folder)
    shutil.rmtree(tmp)



def get_camvid_dataset_parameters(data_path, dataset_type="train", transforms=None):
    """ Get CamVid parameters compatible with DanseData dataset class.

    Arguments:
        data_path (string): path to dataset folder.
        dataset_type (string): dataset type (train, test or val).
        transforms (callable, optional): A function/transform that takes in a sample
            and returns a transformed version.

    Returns:
        dictionary with parameters of CamVid dataset.
    """
    return {
        "data_path":
            os.path.join(data_path, "CamVid"),
        "images_folder":
            dataset_type,
        "masks_folder":
            dataset_type + "annot",
        "num_classes":
            11,
        "transforms":
            transforms,
        "dataset_url":
            "https://github.com/alexgkendall/SegNet-Tutorial.git",
        "dataset_folder":
            "CamVid",
        "class_names": [
            'sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian',
            'bicyclist', 'unlabelled'
        ]
    }


def init_semantic_seg_dataset(data_path:str,images_folder : str,masks_folder : str):

    names = os.listdir(os.path.join(data_path, images_folder))
    
    image_paths = []
    mask_paths = []

    for name in names:
        image_paths.append(os.path.join(data_path,images_folder,name))
        mask_paths.append(os.path.join(data_path,masks_folder,name))

    return image_paths,mask_paths


def draw_semantic_segmentation_samples(dataset, n_samples=3):
    """ Draw samples from semantic segmentation dataset.

    Arguments:
        dataset (iterator): dataset class.
        plt (matplotlib.pyplot): canvas to show samples.
        n_samples (int): number of samples to visualize.
    """
    fig, ax = plt.subplots(nrows=n_samples, ncols=2, sharey=True, figsize=(10, 10))
    for i, sample in enumerate(dataset):
        if i >= n_samples:
            break
        ax[i][0].imshow(sample["image"])
        ax[i][0].set_xlabel("image")
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])

        ax[i][1].imshow(sample["mask"])
        ax[i][1].set_xlabel("mask")
        ax[i][1].set_xticks([])
        ax[i][1].set_yticks([])

    plt.tight_layout()
    plt.gcf().canvas.draw()
    plt.show()
    plt.close(fig)


def draw_semantic_segmentation_batch(images, masks_gt, masks_pred=None, n_samples=3):
    """ Draw batch from semantic segmentation dataset.

    Arguments:
        images (torch.Tensor): batch of images.
        masks_gt (torch.LongTensor): batch of ground-truth masks.
        plt (matplotlib.pyplot): canvas to show samples.
        masks_pred (torch.LongTensor, optional): batch of predicted masks.
        n_samples (int): number of samples to visualize.
    """
    nrows = min(images.size(0), n_samples)
    ncols = 2 if masks_pred is None else 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(10, 10))
    for i in range(nrows):
        img = images[i+11].permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(img, 0, 1)
        ax[i][0].imshow(img)
        ax[i][0].set_xlabel("image")
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        gt_mask = masks_gt[i+11].detach().cpu().numpy()
        ax[i][1].imshow(gt_mask)
        ax[i][1].set_xlabel("ground-truth mask")
        ax[i][1].set_xticks([])
        ax[i][1].set_yticks([])
        if masks_pred is not None:
            pred = masks_pred[i+11].detach().cpu().numpy()
            ax[i][2].imshow(pred)
            ax[i][2].set_xlabel("predicted mask")
            ax[i][2].set_xticks([])
            ax[i][2].set_yticks([])

    plt.tight_layout()
    plt.gcf().canvas.draw()
    plt.show()
    plt.close(fig)


class AverageMeter():
    def __init__(self):
        self.loss = 0
        self.count = 0
    
    def update(self, val):
        self.loss+=val
        self.count+=1

    def avg_loss(self):
        return self.loss/self.count
    

def setup_system(system_config : SystemConfig):
    
    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn_deterministic = system_config.cudnn_deterministic

def patch_configs(epoch_num_to_set=TrainerConfig.num_epochs, batch_size_to_set=DataloaderConfig.batch_size,num_workers_to_set =0):
    """ Patches configs if cuda is not available

    Returns:
        returns patched dataloader_config and trainer_config

    """
    # default experiment params
    num_workers_to_set = DataloaderConfig.num_workers

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size_to_set = 16
        num_workers_to_set = 2
        epoch_num_to_set = 1

    dataloader_config = DataloaderConfig(batch_size=batch_size_to_set, num_workers=num_workers_to_set)
    trainer_config = TrainerConfig(device=device, num_epochs=epoch_num_to_set, stage_progress=True)
    return dataloader_config, trainer_config


