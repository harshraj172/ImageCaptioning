import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.coco_dataset import CocoCaptionDataset
from data.utils import RandomAugment
    
def create_dataset(dataset, tokenizer, processor, config, 
                   device=torch.device('cuda'), min_scale=0.5):
        
    if dataset=='caption_coco':
        train_dataset = CocoCaptionDataset(split='train', tokenizer=tokenizer, processor=processor)
        val_dataset = CocoCaptionDataset(split='validation', tokenizer=tokenizer, processor=processor)
        test_dataset = CocoCaptionDataset(split='test', tokenizer=tokenizer, processor=processor)
        return train_dataset, val_dataset, test_dataset
    else:
        raise NotImplementedError
    
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            # pin_memory=True, #FIXME: proxy for 'RuntimeError: cannot pin 'torch.cuda.LongTensor' only dense CPU tensors can be pinned'
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            multiprocessing_context='spawn', # FIXME: for spawn error
        )              
        loaders.append(loader)
    return loaders    
