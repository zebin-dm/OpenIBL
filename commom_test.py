import torch 
 
subset_indices = torch.randperm(10).long().split(5)
print(subset_indices)