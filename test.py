from trainer.model import SemSeg
import torch

# from experiment import experiment
from trainer.dataset import SemSegDataset
from trainer.utils import(
    setup_system,
    get_camvid_dataset_parameters,
    draw_semantic_segmentation_batch,
    patch_configs
)


from torch.utils.data import DataLoader

from albumentations import Compose,Normalize
from albumentations.pytorch import ToTensorV2



# LOad model
model = SemSeg(11,final_upsample=True)
model.load_state_dict(torch.load("checkpoints/Trainer_best.pth"))
model.eval()


# sample = next(iter(experiment.test_loader))

test_dataset =  SemSegDataset(
                    **get_camvid_dataset_parameters(
                        data_path="data",
                        dataset_type="test",
                        transforms=Compose([
                            Normalize(),
                            ToTensorV2()
                        ])
                
                    )
                )
        
test_loader = DataLoader(test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True
            )

sample = next(iter(test_loader))
print(sample.keys())



device = torch.device("cuda")
# put images on the chosen device

model = model.to(device)
images = sample["image"].to(device)

preds = model(images).softmax(dim=1).argmax(dim=1)


print(preds.shape)
# # visualize the results
draw_semantic_segmentation_batch(images, sample["mask"], preds)