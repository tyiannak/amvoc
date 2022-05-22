import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, n_clusters = 5, kmeans_centers=None):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 64), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 64, 3, padding =1)  
        # conv layer (depth from 64 --> 32), 3x3 kernels
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        # conv layer (depth from 32 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(32, 4, 3, padding=1)

        self.pool = nn.MaxPool2d((2,2), 2)

        self.flatten = nn.Flatten()
        ## decoder layers ##
        # a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 32, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(64, 1, 2, stride=2)

    def forward(self, x, decode = True, clustering = False, kmeans_centers=None):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x) # compressed representation
       

        if self.training and decode:
            ## decode ##
            # add transpose conv layers, with relu activation function
            y = F.relu(self.t_conv1(x))        
            y = F.relu(self.t_conv2(y))
            # output layer (with sigmoid for scaling from 0 to 1)
            y = F.sigmoid(self.t_conv3(y)) 
            if not clustering:
              return x, y

        if self.training and clustering:
            
            x = self.flatten(x)
            dist =torch.cdist(x, kmeans_centers)
            q = ((1+dist.pow(2)).pow(-1))/torch.sum((1+dist.pow(2)).pow(-1),dim=1).view(torch.sum((1+dist.pow(2)).pow(-1),dim=1).shape[0],1)
            if decode:
                return x, y, q
            else:
                return x, q
        return x