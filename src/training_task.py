import argparse
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import os
import utils
import audio_recognize as ar
import audio_process as ap
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from scipy.special import erf, erfc

config_data = utils.load_config("config.json")
ST_WIN = config_data['params']['ST_WIN']
ST_STEP = config_data['params']['ST_STEP']
MIN_VOC_DUR = config_data['params']['MIN_VOC_DUR']
F1 = config_data['params']['F1']
F2 = config_data['params']['F2']
thres = config_data['params']['thres']
factor = config_data['params']['factor']
gamma_1 = config_data['params']['gamma_1']
gamma_2 = config_data['params']['gamma_2']
gamma_3 = config_data['params']['gamma_3']
model_name = config_data['params']['model']
num_ann_per_batch = config_data['params']['num_ann_per_batch']


class ConvAutoencoder(nn.Module):
    def __init__(self, n_clusters = 5, kmeans_centers=None):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 64), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 64, 3, padding =1)  
        # conv layer (depth from 64 --> 32), 3x3 kernels
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        # conv layer (depth from 32 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(32, 8, 3, padding=1)

        self.pool = nn.MaxPool2d((2,2), 2)

        self.flatten = nn.Flatten()
        ## decoder layers ##
        # a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 32, 2, stride=2)
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

def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Amvoc")
    parser.add_argument("-i", "--data_folder", required=True, nargs=None,
                        help="Folder")
    parser.add_argument("-ne", "--num_of_epochs", required=True, nargs=None,
                        help="Parameter")
    parser.add_argument("-s", "--save_model", required=False, nargs=None,
                        help="Condition")
    return parser.parse_args()



def load_spectrograms (filename):
  train_data = []
  print(filename)
  spectrogram, sp_time, sp_freq, fs = ap.get_spectrogram(filename,
                                                    ST_WIN, ST_STEP)
  # These should change depending on the signal's size
  spec_resize_ratio_freq = 4
  spec_resize_ratio_time = 4

  f_low = F1 if F1 < fs / 2.0 else fs / 2.0
  f_high = F2 if F2 < fs / 2.0 else fs / 2.0

  # define feature sequence for vocalization detection
  f1 = np.argmin(np.abs(sp_freq - f_low))
  f2 = np.argmin(np.abs(sp_freq - f_high))

  spectral_energy, mean_values, max_values = ap.prepare_features(spectrogram[:,f1:f2])
  seg_limits, thres_sm = ap.get_syllables(spectral_energy,
                                          mean_values,
                                          max_values,
                                          ST_STEP,
                                          threshold_per=thres * 100,
                                          factor=factor, 
                                          min_duration=MIN_VOC_DUR,
                                          )
  
  train_data += (ar.cluster_syllables(seg_limits, spectrogram,
                                          sp_freq, f_low, f_high,  ST_STEP, train = True))
  return train_data

def PWLoss(batch, outputs, pw_constraints):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return (1/batch)*torch.sum(torch.tensor(pw_constraints).to(device)*torch.cdist(outputs, outputs).pow(2))


def data_prep(spectrogram):

    time_limit = 64
    for i in range(len(spectrogram)):
        
        if len(spectrogram[i])>time_limit:
            spectrogram[i] = spectrogram[i][int((len(spectrogram[i])-time_limit)/2):int((len(spectrogram[i])-time_limit)/2)+time_limit,:]/np.amax(spectrogram[i])
        elif len(spectrogram[i])<time_limit:
            spectrogram[i] = np.pad(spectrogram[i]/np.amax(spectrogram[i]), ((int((time_limit-spectrogram[i].shape[0])/2), (time_limit-spectrogram[i].shape[0]) - int((time_limit-spectrogram[i].shape[0])/2)),(0,0)))
        else:
            spectrogram[i] = spectrogram[i]/np.amax(spectrogram[i])

    spectrogram=np.array(spectrogram, dtype=float)

    spectrogram = spectrogram.reshape(spectrogram.shape[0], 1, spectrogram.shape[1], spectrogram.shape[2])
    dataset = TensorDataset(torch.tensor(spectrogram, dtype = torch.float))

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle = False)
    return train_loader


def plot_func():
    
    locs, _ = plt.yticks()
    locs = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    plt.yticks(locs, ('110', '100', '90' , '80', '70', '60', '50', '40', '30'))
    locs_x = [0, 20, 40, 60, 80, 100, 120, 140]
    plt.xticks(locs_x, ('0', '40', '80', '120', '160', '200', '240', '280'))
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (kHz)')

# def train_clust(spectrogram, train_loader, outputs_init, n_clusters):

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     model = torch.load(model_name, map_location=device)
    
#     criteria = [nn.BCELoss(),nn.KLDivLoss(reduction='batchmean')]

#     # specify loss function
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # number of epochs to train the model
#     n_epochs = 3

#     model = model.float()

#     clusterer = KMeans(n_clusters=n_clusters, random_state=9)

#     y = clusterer.fit_predict(np.array(outputs_init,dtype=object))
#     kmeans_centers = clusterer.cluster_centers_
#     kmeans_tensor = torch.tensor(kmeans_centers).to(device).float()
#     epoch = 0
#     train_loss=0.0
#     end=False

#     pairwise_constraints = np.zeros((len(spectrogram), len(spectrogram)))
#     for epoch in range(n_epochs):
#         if train_loss<-0.1 and epoch>1:
#             # print(epoch)
#             break
#         train_loss = 0.0
#         outputs_ = []
#         answer = 'start'
#         for i,feats in enumerate(train_loader):
#         # monitor training loss
#             if i ==0:
#                 batch_size = feats[0].shape[0]
                
#             batch = feats[0].shape[0]
#             feats[0] = feats[0].to(device)
#             ###################
#             # train the model #
#             ###################
            
#             model.train()
#             # clear the gradients of all optimized variables
#             optimizer.zero_grad()
#             # forward pass: compute predicted outputs by passing inputs to the model
#             outputs, rec, distr = model(feats[0].float(), decode = True, clustering = True, kmeans_centers = torch.tensor(kmeans_tensor).to(device).float())

#             outputs_ += outputs
#             f_j = torch.sum(distr, dim=0)
#             # p distribution
#             p = (distr.pow(2)/f_j.view(1, f_j.shape[0]))/torch.sum(distr.pow(2)/f_j.view(1, f_j.shape[0]), dim=1).view(torch.sum(distr.pow(2)/f_j.view(1, f_j.shape[0]), dim=1).shape[0],1)
#             cnt = 0
            
#             if epoch==0:
#                 while (1):
#                     dist = torch.cdist(outputs, outputs).cpu().detach().numpy()
#                     # keep the largest probability of belonging to a class 
#                     max_ = np.amax(distr.cpu().detach().numpy(), axis=1)
#                     max_ind = np.argmax(distr.cpu().detach().numpy(), axis=1)
#                     sorted_max_ind = np.argsort(max_)
#                     # choose the USV with the greatest uncertainty of belonging to a class
#                     x = sorted_max_ind[cnt]
#                     # x = np.random.randint(feats[0].shape[0])
#                     y = np.random.randint(feats[0].shape[0])
                
#                     if (i*batch+x)<len(pairwise_constraints) and (i*batch+y)<len(pairwise_constraints):
#                         if x==y:
#                             continue
#                         else:
#                             cnt += 1
#                     plt.ion()
#                     plt.show()
#                     if answer!='stop':
#                         plt.figure(figsize=(5,5))
#                         plt.subplot(1,2,1)
#                         plot_func()
#                         plt.imshow(np.flip(spectrogram[i*batch+x], axis=1).T)
#                         plt.subplot(1,2,2)
#                         plot_func()
#                         plt.imshow(np.flip(spectrogram[i*batch+y], axis=1).T)
#                         plt.pause(0.001)
#                         plt.draw()
#                         print()
#                         answer = input("Should the two vocalizations belong to the same cluster? (y/n). If you want to stop, type 'stop'. \n")
#                         plt.close()
                        
#                     while (answer!='stop'):
#                         if answer=='y':

#                             pairwise_constraints[i*batch+x, i*batch+y] = erf(25/dist[x,y])
#                             pairwise_constraints[i*batch+y, i*batch+x] = erf(25/dist[x,y])

#                             break
#                         elif answer=='n':
#                             pairwise_constraints[i*batch+x, i*batch+y] = erf(-25/dist[x,y])
#                             pairwise_constraints[i*batch+y, i*batch+x] = erf(-25/dist[x,y])

#                             break
#                         else:
#                             print()
#                             answer = input("Please provide a new answer (y/n) \n")
#                     if cnt == num_ann_per_batch:
#                         break

#             loss = gamma_1*criteria[0](rec, feats[0].float())+gamma_2*criteria[1](torch.log(distr),p) + gamma_3*(1/(epoch+1))*PWLoss(batch_size, outputs, pairwise_constraints[i*batch:(i+1)*batch, i*batch:(i+1)*batch])
    
#             # backward pass: compute gradient of the loss with respect to model parameters
#             loss.backward()
#             # perform a single optimization step (parameter update)
#             optimizer.step()
#             #update kmeans centers according to gradient descent equation
#             for i in range (kmeans_tensor.shape[0]):
#                 der_loss_centers = -gamma_2*2*torch.sum((((1+(torch.cdist(outputs, kmeans_tensor[i,:].view(1,kmeans_tensor.shape[1]))).pow(2)).pow(-1)).view(batch)*(p[:,i]-distr[:,i]).view(batch))[:,None]*(outputs-kmeans_tensor[i,:]), dim=0)
#                 kmeans_tensor[i,:] = kmeans_tensor[i,:] - 1000* der_loss_centers 
#             train_loss += loss.item()

#         plt.close()

#         train_loss = train_loss/len(train_loader)
#         print('Epoch: {} \tTraining Loss: {:.6f}'.format(
#                 epoch, 
#                 train_loss
#                 ))
#     print("TRAINING IS DONE")
#     return model

def train_clust(model, train_loader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    criteria = [nn.BCELoss(),nn.KLDivLoss(reduction='batchmean')]

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # number of epochs to train the model
    n_epochs = 2

    model = model.float()

    kmeans_centers = np.load('./dash/kmeans_centers.npy')
    kmeans_tensor = torch.tensor(kmeans_centers).to(device).float()
    train_loss=0.0

    pairwise_constraints = np.load('./dash/pwc.npy')
    for epoch in range(n_epochs):
        if train_loss<-0.1 and epoch>0:
            break
        train_loss = 0.0

        for i,feats in enumerate(train_loader):
        # monitor training loss
            if i ==0:
                batch_size = feats[0].shape[0]
                
            batch = feats[0].shape[0]
            feats[0] = feats[0].to(device)
            ###################
            # train the model #
            ###################
            
            model.train()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs, rec, distr = model(feats[0].float(), decode = True, clustering = True, kmeans_centers = torch.tensor(kmeans_centers).to(device).float())

            f_j = torch.sum(distr, dim=0)
            # p distribution
            p = (distr.pow(2)/f_j.view(1, f_j.shape[0]))/torch.sum(distr.pow(2)/f_j.view(1, f_j.shape[0]), dim=1).view(torch.sum(distr.pow(2)/f_j.view(1, f_j.shape[0]), dim=1).shape[0],1)
            cnt = 0

            loss = gamma_1*criteria[0](rec, feats[0].float())+gamma_2*criteria[1](torch.log(distr),p) + gamma_3*(1/(epoch+1))*PWLoss(batch_size, outputs, pairwise_constraints[i*batch:(i+1)*batch, i*batch:(i+1)*batch])
    
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            #update kmeans centers according to gradient descent equation
            for i in range (kmeans_tensor.shape[0]):
                der_loss_centers = -gamma_2*2*torch.sum((((1+(torch.cdist(outputs, kmeans_tensor[i,:].view(1,kmeans_tensor.shape[1]))).pow(2)).pow(-1)).view(batch)*(p[:,i]-distr[:,i]).view(batch))[:,None]*(outputs-kmeans_tensor[i,:]), dim=0)
                kmeans_tensor[i,:] = kmeans_tensor[i,:] - 1000* der_loss_centers 
            train_loss += loss.item()

        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, 
                train_loss
                ))
    print("TRAINING IS DONE")
    return model

def train_one_batch(model, optimizer, batch, pairwise_constraints, kmeans_centers, batch_num):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    criteria = [nn.BCELoss(),nn.KLDivLoss(reduction='batchmean')]

    # specify loss function

    model = model.float()

    batch_size = batch.shape[0]
    batch = batch.to(device)

    ###################
    # train the model #
    ###################
    
    model.train()
    # clear the gradients of all optimized variables
    optimizer.zero_grad()
    # forward pass: compute predicted outputs by passing inputs to the model
    outputs, rec, distr = model(batch.float(), decode = True, clustering = True, kmeans_centers = torch.tensor(kmeans_centers).to(device).float())

    f_j = torch.sum(distr, dim=0)
    # p distribution
    p = (distr.pow(2)/f_j.view(1, f_j.shape[0]))/torch.sum(distr.pow(2)/f_j.view(1, f_j.shape[0]), dim=1).view(torch.sum(distr.pow(2)/f_j.view(1, f_j.shape[0]), dim=1).shape[0],1)

    loss = gamma_1*criteria[0](rec, batch.float())+gamma_2*criteria[1](torch.log(distr),p) + gamma_3*PWLoss(batch_size, outputs, pairwise_constraints[batch_num*batch_size:(batch_num+1)*batch_size, batch_num*batch_size:(batch_num+1)*batch_size])
    
    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # perform a single optimization step (parameter update)
    optimizer.step()
    kmeans_tensor= torch.tensor(kmeans_centers).to(device).float()
    #update kmeans centers according to gradient descent equation
    for i in range (kmeans_tensor.shape[0]):
        der_loss_centers = -gamma_2*2*torch.sum((((1+(torch.cdist(outputs, kmeans_tensor[i,:].view(1,kmeans_tensor.shape[1]))).pow(2)).pow(-1)).view(batch_size)*(p[:,i]-distr[:,i]).view(batch_size))[:,None]*(outputs-kmeans_tensor[i,:]), dim=0)
        kmeans_tensor[i,:] = kmeans_tensor[i,:] - 1000* der_loss_centers 
    kmeans_centers=kmeans_tensor.detach().numpy()
    return model, kmeans_centers, optimizer

def set_constraints(train_loader, model, batch, kmeans_centers, cnt):
    pairwise_constraints=np.load('./dash/pwc.npy')
    model = model.float()
    with torch.no_grad():
        outputs, distr = model(batch, decode=False, clustering=True, kmeans_centers = torch.tensor(kmeans_centers).float())
    
    while (1):
        dist = torch.cdist(outputs, outputs).cpu().detach().numpy()
        # keep the largest probability of belonging to a class 
        max_ = np.amax(distr.cpu().detach().numpy(), axis=1)
        max_ind = np.argmax(distr.cpu().detach().numpy(), axis=1)
        sorted_max_ind = np.argsort(max_)
        # choose the USV with the greatest uncertainty of belonging to a class
        x = sorted_max_ind[cnt]
        # x = np.random.randint(feats[0].shape[0])
        y = np.random.randint(batch.shape[0])
    
        if x<len(pairwise_constraints) and y<len(pairwise_constraints):
            if x==y:
                continue
        return x, y, dist[x,y]

if __name__ == '__main__':

    args = parse_arguments()

    train_data = []
    path = args.data_folder
    # Load spectrograms 
    for folder in os.walk(path):
        for wavfile in folder[2]:
            if wavfile[-3:]=='WAV' or wavfile[-3:]=='wav':
                filename = folder[0] + '/' + wavfile
                train_data += load_spectrograms(filename)
    

    # # Next part is for removing the false negatives from the dataset. 

    # for cur_image in train_data:
    #     if len(cur_image)==0:
    #         train_data.remove(cur_image)
    # test = []
    # # in order to classify the images, we use the following metrics
    # for cur_image in train_data:
    #     test.append([np.mean(cur_image/np.amax(cur_image)),np.var(cur_image/np.amax(cur_image)), np.mean(cur_image-np.amax(cur_image)), np.var(cur_image-np.amax(cur_image))])

    # # clustering to split the two categories
    # clusterer = KMeans(n_clusters=2)
    # y = clusterer.fit_predict(test)
    # train_data = np.array(train_data, dtype=object)
    # # we compare the mean of variances of the images in the two categories
    # var_0, var_1 = [], []
    # for element in train_data[y==1]:
    #     var_1.append(np.var(element))
    # for element in train_data[y==0]:
    #     var_0.append(np.var(element))
    # var_1 = np.mean(var_1)
    # var_0 = np.mean(var_0)
    # # we choose the images with the highest variance (true positives)
    # if var_1 > var_0:
    #     train_data = list(train_data[y==1])
    # else:
    #     train_data= list(train_data[y==0])

    time_limit = 64
    # cropping/padding the images and normalizing between (0,1)
    for i in range(len(train_data)):
        if len(train_data[i])>time_limit:
            train_data[i] = train_data[i][int((len(train_data[i])-time_limit)/2):int((len(train_data[i])-time_limit)/2)+time_limit,:]/np.amax(train_data[i])
        elif len(train_data[i])<time_limit:
            train_data[i] = np.pad(train_data[i]/np.amax(train_data[i]), ((int((time_limit-train_data[i].shape[0])/2), (time_limit-train_data[i].shape[0]) - int((time_limit-train_data[i].shape[0])/2)),(0,0)))
        else:
            train_data[i] = train_data[i]/np.amax(train_data[i])

    train_data=np.array(train_data)

    train_data, test_data = train_test_split(train_data, test_size = 0.2) 
    
    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])
    dataset = TensorDataset(torch.tensor(train_data, dtype = torch.float))

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle = True)
    model = ConvAutoencoder()

    # specify loss function
    criterion = nn.BCELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # number of epochs to train the model
    n_epochs = int(args.num_of_epochs)

    if torch.cuda.is_available():
        model.cuda()
    summary(model, (1, 64, 160))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.float()
    train_loss_acc=[]
    for epoch in range(n_epochs):

        # monitor training loss
        train_loss = 0.0
        
        for feats in train_loader:
        
            feats[0] = feats[0].to(device)
            ###################
            # train the model #
            ###################
            
            # _ stands in for labels, here
            model.train()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            _,outputs = model(feats[0].float(), decode=True)

            loss = criterion(outputs, feats[0].float())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()

            train_loss += loss.item()
            
        train_loss = train_loss/len(train_loader)
        train_loss_acc.append(train_loss)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, 
                train_loss
                ))
        epoch+=1
    
    # plot training loss curve

    # plt.plot(range(1,11),train_loss_acc, '-o')
    # plt.title("Training loss")
    # plt.xlabel("Number of epochs")
    # plt.ylabel("Loss")
    # plt.show()

    # we test the autoencoder on data we didn't use for training

    # test_data = load_spectrograms( "/content/drive/My Drive/B148_test_small.wav")

    # time_limit = 64
    # for i in range(len(test_data)):
        
    #     if len(test_data[i])>time_limit:
    #         test_data[i] = test_data[i][int((len(test_data[i])-time_limit)/2):int((len(test_data[i])-time_limit)/2)+time_limit,:]/np.amax(test_data[i])
    #     elif len(test_data[i])<time_limit:
    #         test_data[i] = np.pad(test_data[i]/np.amax(test_data[i]), ((int((time_limit-test_data[i].shape[0])/2), (time_limit-test_data[i].shape[0]) - int((time_limit-test_data[i].shape[0])/2)),(0,0)))
    #     else:
    #         test_data[i] = test_data[i]/np.amax(test_data[i])

    # test_data=np.array(test_data)
    # test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])
    # test_dataset = TensorDataset(torch.tensor(test_data, dtype = torch.float))

    # batch_size = 32
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = False)
    # outputs = []
    # model.train()
    # with torch.no_grad():
    #     for data in test_loader:
            # _, outputs_ += model(data[0].to(device), decode=True)
            #   outputs+=outputs_

    # for i in range(len(outputs)):
    #     outputs[i] = outputs[i].cpu().detach().numpy()


    # Reconstruction example

    # fig = plt.figure(figsize = (10,10))
    # fig.add_subplot(1,2,1)
    # plt.imshow(test_data[156][0].T)
    # fig.add_subplot(1,2,2)
    # plt.imshow(outputs[156].reshape(outputs[0].shape[1], outputs[0].shape[2]).T)
    # plt.show()

    if args.save_model=="true" or args.save_model=='True' or args.save_model=='1':

        model = model.to("cpu")
        path = "./new_model"
        torch.save(model, path)