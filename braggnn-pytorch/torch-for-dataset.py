import numpy as np
import os
import time
import h5py
import torch
from torch import nn
from model import BraggNN
# from dataset import BraggNNDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


with h5py.File("./valid-ds-psz11-new.h5", 'r') as fp:
    patch = fp['patch'][:]
    ploc  = fp['ploc'][:] - 0.5
    
# device = torch.device("cuda")
input_tensor = torch.from_numpy(patch)
input_tensor = input_tensor.cuda()

iters = 20
psz = 11
model  = BraggNN(imgsz=psz, fcsz=(16, 8, 4, 2)) 
mdl_fn = f'0center-gpu.pth'
model.load_state_dict(torch.load(mdl_fn, map_location=torch.device('cuda')))
model = model.cuda()

batch_size = 16384

chunks = 0
if batch_size < 13799:
    chunks = len(input_tensor) // batch_size
    split_tensor = np.split(patch[:batch_size*chunks], chunks, axis=0)
    for i in range(len(split_tensor)):
        split_tensor[i] = (torch.from_numpy(split_tensor[i])).cuda()
    print("Each chunk in the split tensor has shape: ", split_tensor[0].shape)

last_tensor = patch[batch_size*chunks:]
shape = np.shape(last_tensor)
padded_array = np.zeros((batch_size, 1, 11, 11), dtype=np.float32)
padded_array[:shape[0],:shape[1]] = last_tensor
# last = padded_array

last = torch.from_numpy(padded_array).cuda()
print("Last has shape: ", last.shape)



pt_total_time = 0.0
for i in range(iters):
    
    pred_list = torch.empty((batch_size * (chunks+1), 2), dtype=torch.float32)
    
    pt_start_time = time.time()
    if batch_size < 13799:
        k = 0
        for j in range(chunks):
            #with torch.no_grad():
            pred_list[batch_size * j:batch_size*(j+1)] = model.forward(split_tensor[j]).cpu()
            k = j
        
        #with torch.no_grad():
        pred_list[batch_size*(k+1):batch_size*(k+2)] = model.forward(last).cpu()
    else:
        pred_list = model.forward(last).cpu()
    
    pt_end_time = time.time()
    print(pt_end_time-pt_start_time)
    
    # disregard the first iteration while measuring time
    if i != 0:
        pt_total_time += (pt_end_time - pt_start_time)

# # 95th percentile should be around 0.65 here
l2norm_ml = np.sqrt(np.sum((ploc - ((pred_list.detach().numpy())[:13799]))**2, axis=1)) * 11
print("Error in pixel of %d samples for PT: Avg: %.3f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f" % ((l2norm_ml.shape[0], l2norm_ml.mean(), ) + tuple(np.percentile(l2norm_ml, (50, 75, 95, 99.5))) )) 

print("PT: ", pt_total_time/iters)
# print(pred * 11)