from moviepy.editor import ImageSequenceClip
import os
import pandas as pd
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

import sys, time

from model import PilotNet


class Data():
    def __init__(self, path="data", load=True):
        # find all folders, get data.csv
        data = []
        for folder in os.listdir(path):
            data_csv = os.path.join(path, folder, "data.csv")
            if os.path.exists(data_csv):
                data.append(pd.read_csv(data_csv))

        self.data = pd.concat(data)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.load_from_disk = load

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):   
        image_path, throttle, steer = self.data.iloc[idx]
        if self.load_from_disk:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = image_path
        actuation = torch.tensor([throttle, steer], dtype=torch.float32)

        return image, actuation
    
    def get_tensors(self, test_train_split=0.8):
        split = int(len(self) * test_train_split)
        
        # shuffle
        data = self.data.sample(frac=1)
        train_data = data.iloc[:split]
        val_data = data.iloc[split:]

        train_x = torch.Tensor(len(train_data), 3, 66, 200) # NOTE: hardcoded image size
        train_y = torch.Tensor(len(train_data), 2)
        
        for i, (image_path, throttle, steer) in enumerate(train_data.values):
            image, actuation = self.__getitem__(i)
            train_x[i] = image
            train_y[i] = actuation

        val_x = torch.Tensor(len(val_data), 3, 66, 200) # NOTE: hardcoded image size
        val_y = torch.Tensor(len(val_data), 2)

        for i, (image_path, throttle, steer) in enumerate(val_data.values):
            image, actuation = self.__getitem__(i)
            val_x[i] = image
            val_y[i] = actuation

        return train_x, train_y, val_x, val_y

def data_viz(save_path="data.mp4"):
    # load data
    data = Data(load=False)
    
    images = []
    for i, (image_path, actuation) in enumerate(data): # its only path because we set load=False
        images.append(image_path)

    # create video
    clip = ImageSequenceClip(images, fps=30)
    clip.write_videofile(save_path)

def test_model(model_pth="model.pth", data_path="data"):
    model = PilotNet()
    model.load_state_dict(torch.load(model_pth))
    model.eval()
    data = Data(data_path)

    true_y_steer = []
    true_y_throttle = []
    pred_y_steer = []
    pred_y_throttle = []
    x = []

    t0 = time.perf_counter()
    for i, (image, actuation) in enumerate(data):
        print(f"Processing {i} of {len(data)}", end="\r")
        out = model.forward(image.unsqueeze(0))
        pred_y_steer.append(out[0][1].item())
        pred_y_throttle.append(out[0][0].item())
        true_y_steer.append(actuation[1].item())
        true_y_throttle.append(actuation[0].item())
        x.append(i)
    t1 = time.perf_counter()
    print(f"Inference Time per frame: {(t1-t0)/len(data)}")
    
    # subplot
    fig, axs = plt.subplots(2)
    axs[0].plot(x, true_y_steer, label="True")
    axs[0].plot(x, pred_y_steer, label="Predicted")
    axs[0].set_title("Steer")
    axs[0].legend()

    axs[1].plot(x, true_y_throttle, label="True")
    axs[1].plot(x, pred_y_throttle, label="Predicted")
    axs[1].set_title("Throttle")
    axs[1].legend()

    plt.show()

if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "viz":
        data_viz(*args[1:])
    elif args[0] == "test":
        test_model(*args[1:])
    else:
        raise ValueError("Invalid command")