from torch.utils.data import Dataset
import torch
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def getFileName(path,suffix):
    ## function used to get file names
    NameList=[]
    FileList = os.listdir(path)
    for i in FileList:
        if os.path.splitext(i)[1] == suffix:
            NameList.append(i)
    return NameList

def normalization(data, maxR = 0.99, minR = 40):
    ## normalize data into range (0, 1), input format: tensor
    Imin = data.min()
    Irange = data.max() - Imin
    Imax = Imin + Irange * maxR
    Imin = Imin + Irange * minR
    data = (data - Imin) / (Imax - Imin)
    data.clamp_(0.0, 1.0)
    return data

def reshapeTimeStep(pos, neg, timeStep):
    interval = int(pos.shape[0] / timeStep)
    posNew = np.zeros((timeStep, pos.shape[1], pos.shape[2]))
    negNew = np.zeros((timeStep, neg.shape[1], neg.shape[2]))
    for i in range(timeStep):
        posNew[i,:] = pos[i*interval:(i+1)*interval,:].sum(0)
        negNew[i,:] = neg[i*interval:(i+1)*interval,:].sum(0)
    return posNew, negNew
class Dataset_EFNet(Dataset):
    def __init__(self, dataset_opt, norm=False, timeStep=30):
        ## EventDir: directory path of event data for train
        ## FrameDir: directory path of frame data
        ## EframeDir: directory path of E→F data
        ## GTDir: directory path of APS data for train
        self.EventDir = dataset_opt['dataroot_Event']
        self.GTDir = dataset_opt['dataroot_GT']
        self.FrameDir = dataset_opt['dataroot_Frame']
        self.EframeDir = dataset_opt['dataroot_EFrame']
        self.timeStep = timeStep
        self.norm = norm
        self.EventNames = getFileName(self.EventDir, '.npy')
        self.GTNames = getFileName(self.GTDir, '.png')
        self.FrameNames = getFileName(self.FrameDir, '.npy')
        self.EframeNames = getFileName(self.EframeDir, '.npy')
        self.EventNames.sort()
        self.GTNames.sort()
        self.FrameNames.sort()
        self.EframeNames.sort()

    def __len__(self):
        assert len(self.EventNames) == len(self.GTNames)
        assert len(self.EventNames) == len(self.FrameNames)
        assert len(self.EventNames) == len(self.EframeNames)
        return len(self.EventNames)

    def __getitem__(self, index):

        EventName = self.EventNames[index]
        GTName = self.GTNames[index]
        FrameName = self.FrameNames[index]
        EframeName = self.EframeNames[index]
        ## warping event data
        EventPath = os.path.join(self.EventDir, EventName)
        EventData = np.load(EventPath, allow_pickle=True).item()
        pos = EventData['Pos']
        neg = EventData['Neg']
        assert pos.shape[0] % self.timeStep == 0, "Inappropriate time step"
        if (pos.shape[0] != self.timeStep):
            pos, neg = reshapeTimeStep(pos, neg, self.timeStep)
        pos = np.expand_dims(pos, axis=1)
        neg = np.expand_dims(neg, axis=1)
        EventInput = torch.FloatTensor(np.concatenate((pos, neg), axis=1))  ## EventInput = (step, channel, H, W)

        ## warping aps data
        GTPath = os.path.join(self.GTDir, GTName)
        GTInput = plt.imread(GTPath, plt.cm.gray)
        if GTInput.ndim == 3:  # get 1 dim image
            GTInput = GTInput[:, :, 0]
        GTInput = torch.FloatTensor(np.expand_dims(GTInput, axis=0))

        FramePath = os.path.join(self.FrameDir, FrameName)
        FrameData = np.load(FramePath)
        EframePath = os.path.join(self.EframeDir, EframeName)
        EframeData = np.load(EframePath)
        FrameInput = torch.FloatTensor(FrameData)
        EframeInput = torch.FloatTensor(EframeData)
        if self.norm:
            EventInput = normalization(EventInput)
            GTInput = normalization(GTInput)
            FrameInput = normalization(FrameInput)
            EframeInput = normalization(EframeInput)

        return {'Event':EventInput, 'Frame':FrameInput, 'EFrame':EframeInput, 'GT': GTInput, 'Frame_path':self.FrameDir}