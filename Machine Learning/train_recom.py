import argparse

from mmcv import Config

from mmfashion.apis import (get_root_logger, init_dist, set_random_seed, train_fashion_recommender)
from mmfashion.datasets import build_dataset
from mmfashion.models import build_fashion_recommender
from mmfashion.utils import init_weights_from

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a Fashion Attribute Predictor'
    )
