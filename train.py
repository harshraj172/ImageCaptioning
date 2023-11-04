'''
https://github.com/salesforce/BLIP/blob/main/train_caption.py
'''

import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import wandb 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from models import base_decoder, model2_decoder, model3_decoder
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval

def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 1000

    for i, (samples, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples['pixel_values'] = samples['pixel_values'].to(device)
        samples['input_ids'] = samples['input_ids'].to(device)
        loss = model(samples)      
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        # wandb.log({'Train Loss': loss.item()})
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # checkpointing after epoch
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, checkpoint  


@torch.no_grad()
def evaluate(model, eval_model, data_loader, device, config):
    # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result, all_scores = [], []
    for samples, ref_captions, image_ids in metric_logger.log_every(data_loader, print_freq, header): 
        samples['pixel_values'] = samples['pixel_values'].to(device)
        samples['input_ids'] = samples['input_ids'].to(device)
        captions = model.generate(samples, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                  min_length=config['min_length'])
        
        for ref_caption, caption, eval_pixel_value, img_id in zip(ref_captions, captions, samples['eval_pixel_values'], image_ids):
            image_feature = eval_model.encode_img(eval_pixel_value)
            ref_caption_feature = eval_model.encode_text(ref_caption)
            caption_feature = eval_model.encode_text(caption)
            image_feature = F.normalize(image_feature, dim=-1)
            ref_caption_feature = F.normalize(ref_caption_feature, dim=-1)
            caption_feature = F.normalize(caption_feature, dim=-1)
            
            text_feature = torch.cat(caption_feature, ref_caption_feature, axis=1)
            log_probs = torch.sigmoid(image_feature @ text_feature.T * model.logit_scale.exp() + model.logit_bias)
            
            all_scores.append(1 if log_probs[0]>log_probs[1] else 0)
            result.append({"image_id": img_id.item(), "ref_caption":ref_caption, 
                           "predicted_caption": caption, "caption-log_probs": log_probs[0],
                           "ref_caption-log_probs": log_probs[1], "score": 1 if log_probs[0]>log_probs[1] else 0})
        
    avg_score = sum(all_scores)/len(all_scores)
    # wandb.log({'eval_score': avg_score})
    print('************eval score=', avg_score)
    return result


def main(args, config):
    utils.init_distributed_mode(args)    

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    tokenizer = AutoTokenizer.from_pretrained(config['lm_path'])
    processor = AutoImageProcessor.from_pretrained(config['vision_model_path'])
    eval_model, eval_processor = create_model_from_pretrained(config['eval_model'])
    train_dataset, val_dataset, test_dataset = create_dataset('caption_coco', tokenizer, processor, 
                                                              eval_processor, config, device=device)  
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[4,4,4],
                                                          is_trains=[True, False, False], collate_fns=[None,None,None])         

    #### Model #### 
    print("Creating model")
    # model = base_decoder(pretrained=config['pretrained'], config=config, vision_model_path=config['vision_model_path'], lm_path=config['lm_path'])
    model = model2_decoder(pretrained=config['pretrained'], config=config, vision_model_path=config['vision_model_path'], lm_path=config['lm_path'])
    # model = model3_decoder(pretrained=config['pretrained'], config=config, vision_model_path=config['vision_model_path'], lm_path=config['lm_path'])

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    params_to_optimize = (p for p in list(model.parameters()) if p.requires_grad)
    optimizer = torch.optim.AdamW(params=params_to_optimize, lr=config['init_lr'], weight_decay=config['weight_decay'])
            
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                
            train_stats, checkpoint = train(model, train_loader, optimizer, epoch, device) 
        
        if (epoch+1) % 1 == 0 or epoch == config['max_epoch']-1:
            val_result = evaluate(model_without_ddp, eval_model, val_loader, device, config)  
            val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='image_id')        

            # test_result = evaluate(model_without_ddp, eval_model, test_loader, device, config)  
            # test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d'%epoch, remove_duplicate='image_id')   
            
            # save
            checkpoint_filename = f'model_checkpoint_epoch_{epoch+1}.pth'
            checkpoint_dir = Path(args.output_dir)
            torch.save(checkpoint, checkpoint_dir / checkpoint_filename)
            
        if args.evaluate: 
            break
        dist.barrier()     

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_coco.yaml')
    parser.add_argument('--output_dir', default='output/Caption_coco')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    print('='*100)
    print(config)
    print('='*100)
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.safe_dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    # wandb.init(entity=config['wandb_team'], project=config['wandb_project'], config=config)
    
    main(args, config)
    # wandb.finish()