## Semantic_Segmentation_Using_RESNET18_Encoder_FPN_Decoder 
This repository includes the code to do semantic segmentation of road scenses using Camvid dataset.

![segmentation_result](https://github.com/sairampolina/Roadscenes_Semantic_Segmentation_FPN_Decoder/assets/48856345/00fc1450-7e4c-4f86-a000-1ea3f5fcb605)

## Note:
The resulting model obtained is not the optimal model.It has been trained just for few epochs. THe emphasis is laid on DataLoaders,and Training pipiline rather than results. Feelfree to start from the checkpoint and fine-tune the model.

## TO RUN
1.Clone this repository:

```
git clone --recurse-submodules https://github.com/sairampolina/Roadscenes_Semantic_Segmentation_FPN_Decoder.git
```
2.Setup a conda environment and Install Dependencies

```
 conda create --name <env_name> --file requirements.txt
```
3.To auto-download data and plot loss and metric curves RUN (from root of the directory)

```
python3 experiment.py
```

4.For inference on few test samples RUN:
```
python3 test.py
```
