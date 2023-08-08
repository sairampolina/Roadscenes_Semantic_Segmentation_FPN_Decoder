
from operator import itemgetter

import matplotlib.pyplot as plt
import math

import numpy as np



def plot_iou_eachclass(num_classes: int , metrics : dict):

  
    plt.rcParams['figure.figsize'] = (20,15)
    plt.style.use('ggplot')
    plt.figure()

    rows = len(metrics["epoch"])
    cols  = num_classes

    epochs_ious = np.ndarray(shape=(rows,cols),dtype=np.float16)
    get_iou = itemgetter("iou")

    for idx, metric_dict in enumerate(metrics["test_metric"]):
        epochs_ious[idx,:] = get_iou(metric_dict)

    plot_rows = math.ceil(num_classes/3)
    plot_cols =  3
    # plot each col from epochs_ious array
    for i in range(num_classes):
        plt.subplot(plot_rows,plot_cols,i+1)
        plt.plot(metrics["epoch"], epochs_ious[:,i])

        plt.xlabel("Epochs")
        plt.ylabel("iou_cls"+ str(i))

    plt.show()


def plot_loss_and_metric_curves(metrics:dict):
    plt.rcParams['figure.figsize'] = (8,5)
    plt.style.use('ggplot')

    # create new figure
    plt.figure()
    # plot test and train loss
    plt.plot(metrics["test_loss"], label="test")
    plt.plot(metrics["train_loss"], label="train")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.show()

    # create new figure
    plt.figure()
    # plot mean iou metric for all classes
    plt.plot([metric['miou'] for metric in metrics["test_metric"]], label="miou")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.title('Validation mIoU')
    plt.show()

