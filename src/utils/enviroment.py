import os
import torch

from torch.distributed import init_process_group, destroy_process_group


class DistributedContext(object):
    def __init__(self, configuration):
        """
        sets the distributed environment backend
        torch.distributed.init_process_group(backend='nccl')

        gets the process rank
        os.environ["LOCAL_RANK"]

        destroys all process at exit
        torch.distributed.destroy_process_group(
        """

        self.configuration = configuration
        if self.configuration.environment.distributed:
            init_process_group(backend=self.configuration.environment.backend)

    def __enter__(self):
        if 'LOCAL_RANK' in os.environ:
            self.configuration.trainer.gpu_id = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.configuration.trainer.gpu_id)
            # torch.cuda.device(self.configuration.trainer.gpu_id)
        else:
            self.configuration.trainer.gpu_id = 0
            print('[LOCAL_RANK] is not present in environment')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.configuration.environment.distributed:
            destroy_process_group()
