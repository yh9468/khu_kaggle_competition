from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

class DatasetMNIST(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

if __name__ == "__main__":
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = DatasetMNIST("./fashion-mnist_train.csv", transform=transform)
    loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    for img, label in loader:
        """
        # Fashion MNIST visualize code
        img = transforms.ToPILImage()(img[0])
        img = np.array(img)
        img = np.reshape(img,(28,28))
        plt.imshow(img, cmap="gray")
        plt.show() 
        """

        