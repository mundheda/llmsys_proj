from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import math

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

# ASSIGNMENT 4.2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN SOLUTION
    schedules = [[] for _ in range(num_batches+num_partitions-1)]
    for bidx in range(num_batches):
        for pidx in range(num_partitions):
            job = (bidx, pidx)
            schedules[bidx+pidx].append(job)
    return schedules
    # END SOLUTION

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    # ASSIGNMENT 4.2
    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN SOLUTION
        import math
        num_microbatches = math.ceil(len(x) / self.split_size)
        schedule = _clock_cycles(num_microbatches, len(self.partitions))
        y = self.compute(x, schedule)
        y = torch.concat(y, dim=0)
        return y
        # END SOLUTION

    # ASSIGNMENT 4.2
    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices
        
        # BEGIN SOLUTION
        out_list = []
        output_len = 0
        wait_len = 0
        
        def worker(partition, inp):
            return lambda: partition(inp)
                
        for jobs_list in schedule:
            for (bidx, pidx) in jobs_list:
                
                if pidx == 0:
                    x = batches[bidx*self.split_size:(bidx+1)*self.split_size]
                    wait_len += len(x)
                else:
                    x = self.out_queues[pidx-1].get()
                    x = x[1][1]
                    
                if not isinstance(x, tuple):
                    x = x.to(devices[pidx])
                
                task = Task(worker(partitions[pidx], x))
                self.in_queues[pidx].put(task)                    
                        
        while(output_len < wait_len):
            x = self.out_queues[len(partitions)-1].get()
            out_list.append(x[1][1])
            output_len += len(out_list[-1])
        
        return out_list   
        # END SOLUTION

