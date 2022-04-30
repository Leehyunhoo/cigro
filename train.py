import os
import argparse
import time
import yaml
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from google.cloud import storage
import shutil

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader 
from timm.utils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import time


from .model.model import *
from .dataset import *
from .config import *
from .loss import *
from .processing import *
from .evaluate import *

'''
def upload_metric_result(path_to_file):
    key_name = glob.glob('/home/ubuntu/project/auth-key/*.json')[0]
    storage_client = storage.Client.from_service_account_json(key_name)

    blob_name = "text_folder/metric_result.jpg"
    bucket_name = "nnc-gifticon"

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_file)
    '''
    
    #접근권한 public 으로 설정
    #blob.make_public()
    
    #파일 url 만들어주기
    #url = blob.public_url
    #returns a public url
    #return url

def get_clip_parameters(model, exclude_head=False):
    if exclude_head:
        # FIXME this a bit of a quick and dirty hack to skip classifier head params
        return [p for n, p in model.named_parameters() if 'predict' not in n]
    else:
        return model.parameters()

def train_epoch(
        epoch, model, loader, loss_fn, optimizer, args,
        lr_scheduler=None, output_dir=''):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    clip_params = get_clip_parameters(model, exclude_head='agc' in args.clip_mode)
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(tqdm(loader)):
        
        '''
        input : [tensor, tensor]
        '''
        
        input, target = [x.to(args.device) for x in input], target.to(args.device)
        
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        output = model(input)   
             
        loss = loss_fn(output, target.reshape(-1))
        losses_m.update(loss.item(), input[0].size(0))
        
        optimizer.zero_grad()
        loss.backward()
        
        if args.clip_grad is not None:
            dispatch_clip_grad(clip_params, value = args.clip_grad, mode = args.clip_mode)
            
        optimizer.step()
        
        if args.device != 'cpu':
            torch.cuda.synchronize()
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            logging.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'LR: {lr:.3e}  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input[0].size(0) * args.world_size / batch_time_m.val,
                    rate_avg=input[0].size(0) * args.world_size / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m))

            """if args.save_images and output_dir:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                    padding=0,
                    normalize=True)"""
                    

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, loss_fn, postprocess, args, evaluator=None, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    
    preds, labels = [], []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tqdm(loader)):
            
            input, target = [x.to(args.device) for x in input], target.to(args.device)
            
            last_batch = batch_idx == last_idx
     
            output = model(input)
            loss = loss_fn(output, target.reshape(-1))
            reduced_loss = loss.data
            
            if args.device != 'cpu':
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input[0].size(0))
            
            pred = output.argmax(1).detach().cpu().tolist()
            label = target.reshape(-1).detach().cpu().tolist()
            
            preds.extend(pred)
            labels.extend(label)
            
            
            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m, loss=losses_m))
         
    metric_score =  Evaluatation(args, preds, labels)      
    metric_score['loss'] = losses_m.avg
    return metric_score

def Train(args):
    model = Model(args)
    model.to(args.device)
    loss_fn = ClassificationLoss(args)#DetectionLoss(Config.model_config)
    optimizer = create_optimizer(args, model)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    evaluator = None
    #input_dir = "/content/detection_test/upload_files"
    output_dir = ''
    output_base = './output'
    if os.path.isdir(output_base):
        shutil.rmtree(output_base)
    exp_name = ''#'-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), args.model])
    output_dir = get_outdir(output_base, 'train', exp_name)
    decreasing = True if eval_metric == 'loss' else False
    saver = CheckpointSaver(model, optimizer, args=args,
        checkpoint_dir=output_dir, decreasing=decreasing)
    
    saver.extension = '.pth'
    try:
        ann_files = pd.read_csv(args.file_url, encoding='cp949')
    except:
        try:
            ann_files = pd.read_csv(args.file_url)
        except:
            raise Exception('this file format is not supported...')  
   
    train_ann_files, val_ann_files = PreProcessing(args, ann_files)
    
    dataset_train = Dataset(args, train_ann_files[:args.train_num], trainTransform)
    dataset_eval = Dataset(args, val_ann_files[:args.val_num], valTransform)

    loader_train = data.DataLoader(dataset_train, batch_size = args.batch_size, shuffle = True, drop_last = True)
    loader_eval = data.DataLoader(dataset_eval, batch_size = args.batch_size, shuffle = True, drop_last = False)
    
    patience = 0
    best = 0.0
    
    results = []
    for epoch in range(num_epochs):
        train_metrics = train_epoch(
            epoch, model, loader_train, loss_fn, optimizer, args,
            lr_scheduler=lr_scheduler, output_dir=output_dir)

        eval_metrics = validate(model, loader_eval, loss_fn, None, args, evaluator)

        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
            
        if saver is not None:
            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

            best_metric, best_epoch = saver.save_checkpoint(epoch=epoch, metric=eval_metrics[eval_metric])
            print(best_metric, best_epoch)
            
        '''
        # metric plot
        results.append(eval_metrics[eval_metric])
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.plot(np.arange(1, epoch+2), results)
        plt.savefig('/home/ubuntu/project/deepLearning/acc-plot.jpg')
        upload_metric_result('/home/ubuntu/project/deepLearning/acc-plot.jpg')
        '''
        
        # early stopping
        if best == best_metric: patience +=1
        else: best = best_metric; patience = 0
        
        if patience >= args.early_stopping:
            print('there is no improvement in metric, so training process is terminated...')
            break
    return best_metric


    
    