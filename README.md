## Semantic_Segmentation_Using_RESNET18_Encoder_FPN_Decoder 
This repository includes the code to do END-TO-END semantic segmentation of road scenses using Camvid dataset.

## Note:
* The resulting model obtained is not the optimal model.It has been trained just for few epochs. 
* The emphasis is laid on **End-to End semantic segmenation** pipeline (DataLoaders,and Training pipiline) rather than results. 
* Feelfree to start from the checkpoint and fine-tune the model.

![train_val_loss](https://github.com/sairampolina/Roadscenes_Semantic_Segmentation_FPN_Decoder/assets/48856345/9821b64f-452b-4bb0-9804-2fab0a2e5a69)
![mIoU](https://github.com/sairampolina/Roadscenes_Semantic_Segmentation_FPN_Decoder/assets/48856345/90b415b3-2b13-4406-94fe-b00f1bde9dce)


![results](https://github.com/sairampolina/Roadscenes_Semantic_Segmentation_FPN_Decoder/assets/48856345/db572031-60cf-44b4-b4d8-1ff2006625a0)


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
