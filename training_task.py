import sys
import argparse
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os, glob
import audio_recognize as ar
import audio_process as ap
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# These values are selected based on the evaluation.py script and the results
# obtained from running these scripts on two long annotated recordings
ST_WIN = 0.002    # short-term window
ST_STEP = 0.002   # short-term step
MIN_VOC_DUR = 0.005

# The frequencies used for spectral energy calculation (Hz)
F1 = 30000
F2 = 110000

# Also, default thres value is set to 1.3 (this is the optimal based on
# the same evaluation that led to the parameter set of the
# short-term window and step
thres = 1.

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 64), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 64, 3, padding =1)  
        # conv layer (depth from 64 --> 32), 3x3 kernels
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        # conv layer (depth from 32 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(32, 8, 3, padding=1)

        self.pool = nn.MaxPool2d((2,2), 2)

        ## decoder layers ##
        # a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 32, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(64, 1, 2, stride=2)

    def forward(self, x):
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

        if self.training:
            ## decode ##
            # add transpose conv layers, with relu activation function
            x = F.relu(self.t_conv1(x))        
            x = F.relu(self.t_conv2(x))
            # output layer (with sigmoid for scaling from 0 to 1)
            x = F.sigmoid(self.t_conv3(x))      
        return x

def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Amvoc")
    parser.add_argument("-i", "--data_folder", required=True, nargs=None,
                        help="Folder")
    parser.add_argument("-ne", "--num_of_epochs", required=True, nargs=None,
                        help="Parameter")
    parser.add_argument("-s", "--save_model", required=True, nargs=None,
                        help="Condition")
    return parser.parse_args()



def load_spectrograms (filename):
  train_data = []
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

  spectral_energy_1 = spectrogram.sum(axis=1)
  spectral_energy_2 = spectrogram[:, f1:f2].sum(axis=1)
  seg_limits, thres_sm, _ = ap.get_syllables(spectral_energy_2,
                                        spectral_energy_1,
                                        ST_STEP,
                                        threshold_per=thres * 100,
                                        min_duration=MIN_VOC_DUR)
  
  train_data += (ar.cluster_syllables(seg_limits, spectrogram,
                                          sp_freq, f_low, f_high,  ST_STEP, train = True))
  return train_data

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
    

    # Next part is for removing the false negatives from the dataset. 

    for cur_image in train_data:
        if len(cur_image)==0:
            train_data.remove(cur_image)
    test = []
    # in order to classify the images, we use the following metrics
    for cur_image in train_data:
        test.append([np.mean(cur_image/np.amax(cur_image)),np.var(cur_image), np.mean(cur_image-np.amax(cur_image))])

    # clustering to split the two categories
    clusterer = KMeans(n_clusters=2)
    y = clusterer.fit_predict(test)
    train_data = np.array(train_data, dtype=object)
    # we compare the mean of variances of the images in the two categories
    var_0, var_1 = [], []
    for element in train_data[y==1]:
        var_1.append(np.var(element))
    for element in train_data[y==0]:
        var_0.append(np.var(element))
    var_1 = np.mean(var_1)
    var_0 = np.mean(var_0)
    # we choose the images with the highest variance (true positives)
    if var_1 > var_0:
        train_data = list(train_data[y==1])
    else:
        train_data= list(train_data[y==0])

    # # histogram of durations (in samples) of all the spectrograms
    # # based on that, we choose the width of the spectrograms to be 64 samples (it has to be multiple of 4 because of the 3 max pooling layers)
    # duration = []
    # for image in train_data:
    #     duration.append(image.shape[0])
    # plt.hist(duration, bins = range(min(duration), max(duration)))
    # plt.show()

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

    for epoch in range(n_epochs):

        for feats in train_loader:
        # monitor training loss
            train_loss = 0.0

            feats[0] = feats[0].to(device)
            ###################
            # train the model #
            ###################
            
            # _ stands in for labels, here
            model.train()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(feats[0].float())

            loss = criterion(outputs, feats[0].float())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()

            train_loss += loss.item()
            
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, 
                train_loss
                ))

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
    test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])
    test_dataset = TensorDataset(torch.tensor(test_data, dtype = torch.float))

    batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = False)
    outputs = []
    model.train()
    with torch.no_grad():
        for data in test_loader:
            outputs += model(data[0].to(device))


    for i in range(len(outputs)):
        outputs[i] = outputs[i].cpu().detach().numpy()


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