from torch.utils.data import DataLoader #convert to tensorflow

from .utils import build_from_cfg
from .wrapper import


def build_dataloader(dataset, imgs_per_gpu,
                     num_gpus, drop_last=False,
                     dist=False, **kwargs):
    shuffle = kwargs.get('shuffle', True)
    batch_size = num_gpus * imgs_per_gpu
    num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    return data_loader

def _concat_dataset(cfgs):
    datasets = []
    for cfg in cfgs:
        datasets.append(build_from_cfg(cfg, DATASETS))
    return ConcatDataset(datasets)

def build_dataset(cfg):
    if isinstance(cfg, (list, tuple)):
        dataset = _concat_dataset(cfg)
    else:
        dataset = build_from_cfg(cfg, DATASETS)
    return dataset
