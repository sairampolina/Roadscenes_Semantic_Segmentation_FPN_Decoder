
from tqdm.auto import tqdm
from .utils import AverageMeter

from operator import itemgetter

import torch

def train_hook_default(
        model,
        train_loader,
        loss_fn,
        optimizer,
        data_getter,
        target_getter,
        device,
        stage_progress,
        prefix):
    
    model.train()
    iterator = tqdm(train_loader,disable= not stage_progress, dynamic_ncols=True)
    avg_meter = AverageMeter()

    for i,sample in enumerate(iterator):

        inputs = data_getter(sample).to(device)
        targets = target_getter(sample).to(device)

        preds = model(inputs)
        loss = loss_fn(preds,targets)

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        avg_meter.update(loss.item())

        status = f"{prefix}[Train:{i}] Loss_avg: {avg_meter.avg_loss():.5f} LR: {optimizer.param_groups[0]['lr']}"

        iterator.set_description(status)

    return {'loss': avg_meter.avg_loss()}

        
def test_hook_default(
        model,
        test_loader,
        loss_fn,
        metric_fn,
        optimizer,
        data_getter,
        target_getter,
        device,
        stage_progress,
        get_key_metric = itemgetter("miou"),
        prefix = "",
):
    
    model.eval()

    iterator = tqdm(test_loader, disable= not stage_progress,dynamic_ncols=True)
    avg_meter = AverageMeter()
    metric_fn.reset()

    for i,sample in enumerate(iterator):

        inputs = data_getter(sample).to(device)
        targets = target_getter(sample).to(device)

        with torch.no_grad():
            preds = model(inputs)
            loss = loss_fn(preds, targets)
        avg_meter.update(loss.item())

        preds = preds.softmax(dim=1).detach()

        metric_fn.update_values(preds,targets)

        status = f"{prefix} [Test][{i} Loss_avg : {avg_meter.avg_loss():.5f} mIou : {get_key_metric(metric_fn.get_metric_value()):.5f}"
        iterator.set_description(status)

    output = {"metric": metric_fn.get_metric_value(),"loss": avg_meter.avg_loss()}

    return output


    