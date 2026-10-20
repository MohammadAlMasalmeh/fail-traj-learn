import sys, time, torch
sys.path.insert(0,'scripts/distill')
from model import *
t=time.time()
m=load_model(trainable=True, grad_ckpt=True)
print('load',time.time()-t, torch.cuda.memory_allocated()/1e9)
m.print_trainable_parameters()
