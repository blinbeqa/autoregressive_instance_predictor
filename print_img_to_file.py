import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.image as imgblin

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
#hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
 #                                          transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(np.load('mnist_test_seq.npy'),
                                             batch_size=20, shuffle=True,
                                             num_workers=4)
X = np.load('mnist_test_seq.npy', mmap_mode='r')
print(X[1,1,:,:])

for i in range (len (X[:,1,:,:])):

  imgblin.imsave('name'+ str(i) +'.png', X[i,1,:,:])