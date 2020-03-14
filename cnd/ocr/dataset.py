import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch.utils.data import Dataset



class OcrDataset(Dataset):
    def __init__(self, data_path, csv_file_path, transforms=None):
        self.data = pd.read_csv(csv_file_path)
        self.imgs = self.data['filename'].tolist()
        self.target = self.data['target'].tolist()
        self.transforms = transforms
        
        self.data_path = data_path
        if (self.data_path[-1] != '/'):
            self.data_path += '/'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = plt.imread(self.data_path + self.imgs[idx])
        
        
        text = self.target[idx]
        
        if (self.transforms):
            img = self.transforms(img)

        return {"image": img,
                "targets": text}
