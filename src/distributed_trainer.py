import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedTrainer:
    def __init__(self, config):
        self.config = config
        self.rank = int(os.getenv("SLURM_PROCID", 0))
        dist.init_process_group("nccl", rank=self.rank, world_size=1)
        
    def setup_device(self):
        torch.cuda.set_device(self.rank)
        
    def cleanup(self):
        dist.destroy_process_group()

