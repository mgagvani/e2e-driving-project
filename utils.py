from moviepy import ImageSequenceClip
import os
import pandas as pd
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

import sys, time

from models import PilotNet

plt.switch_backend('Agg')

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
    
    def __getitem__(self, idx_any):   
        idx = int(idx_any)
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
    
    def train_val_split(self, test_train_split=0.8):
        """
        Returns train and validation datasets.
        Used with torch.utils.data.DataLoader
        e.g.
        train_dataset, val_dataset = data.train_val_split()
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        """
        split = int(len(self) * test_train_split)
        indices = torch.randperm(len(self))
        train_indices = indices[:split]
        val_indices = indices[split:]
        
        from torch.utils.data import Subset
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        
        return train_dataset, val_dataset
    
    def get_tensors(self, test_train_split=0.8):
        split = int(len(self) * test_train_split)
        
        # get shuffled indices
        indices = torch.randperm(len(self))
        train_indices = indices[:split]
        val_indices = indices[split:]

        # cuda tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_x = torch.Tensor(len(train_indices), 3, 66, 200) # NOTE: hardcoded image size
        train_y = torch.Tensor(len(train_indices), 2)
        
        # NOTE: there may be a bug here. if we are __getitem__(i),
        # then, we are doing self.data.iloc[i] which is not shuffled
        # so, we might as well not have shuffled in the first place
        # this could be fixed by getting indices and shuffling them.
        for i, idx in enumerate(train_indices):
            if i % 1000 == 0:
                print(f"Loading train {i} of {len(train_indices)}", end="\r")
            image, actuation = self.__getitem__(int(idx))
            train_x[i] = image
            train_y[i] = actuation

        val_x = torch.Tensor(len(val_indices), 3, 66, 200) # NOTE: hardcoded image size
        val_y = torch.Tensor(len(val_indices), 2)

        for i, idx in enumerate(val_indices):
            if i % 1000 == 0:
                print(f"Loading val {i} of {len(val_indices)}", end="\r")
            image, actuation = self.__getitem__(int(idx))
            val_x[i] = image
            val_y[i] = actuation

        # cuda
        new_tensors = []
        for tensor in [train_x, train_y, val_x, val_y]:
            new_tensors.append(tensor.to(device))

        return new_tensors

def data_viz(save_path="data.mp4"):
    # load data
    data = Data(load=False)
    
    images = []
    for i, (image_path, actuation) in enumerate(data): # its only path because we set load=False
        images.append(image_path)

    # create video
    clip = ImageSequenceClip(images, fps=30)
    clip.write_videofile(save_path)

def test_model(model_pth, data_pth):
    model = PilotNet()
    model.load_state_dict(torch.load(model_pth))
    model.eval()
    data = Data(data_pth)

    true_y_steer = []
    true_y_throttle = []
    pred_y_steer = []
    pred_y_throttle = []
    x = []

    BATCH_SIZE = 8192
    chunk_ends = [i for i in range(0, len(data), BATCH_SIZE)]
    chunk_ends.append(len(data))

    chunks = [i for i in zip(chunk_ends[:-1], chunk_ends[1:])]

    idx_buckets = [list(range(a, b)) for a, b in chunks]

    t0 = time.perf_counter()
    for bucket in tqdm(idx_buckets, desc="Inference"):
        images = []
        actuations = []
        for i in bucket:
            image, actuation = data[i]
            images.append(image)
            actuations.append(actuation)
        images = torch.stack(images)
        actuations = torch.stack(actuations)
        # old code: actually inference
        out = model.forward(images)
        for i, out in enumerate(out):
            pred_y_steer.append(out[1].item())
            pred_y_throttle.append(out[0].item())
            true_y_steer.append(actuations[i][1].item())
            true_y_throttle.append(actuations[i][0].item())
            x.append(i + bucket[0])
    t1 = time.perf_counter()
    print(f"Inference Time per frame: {(t1-t0)/len(data)}")

    plt.cla()
    
    # subplot
    fig, axs = plt.subplots(2, figsize=(12, 8))
    axs[0].plot(x, true_y_steer, label="True", linewidth=0.5)
    axs[0].plot(x, pred_y_steer, label="Predicted", linewidth=0.5)
    axs[0].set_title("Steer")
    axs[0].set_ylim([-1.5, 1.5])
    axs[0].legend()

    axs[1].plot(x, true_y_throttle, label="True", linewidth=0.5)
    axs[1].plot(x, pred_y_throttle, label="Predicted", linewidth=0.5)
    axs[1].set_title("Throttle")
    axs[1].legend()

    plt.savefig("eval_plot.png", dpi=900)

    # video showing the graph zoomed in on the x-axis scrolling through the data
    images = []
    HORIZONTAL = H = 200
    STEP = 10
    for i in range(0, len(x)-H, STEP):
        print(f"Frame {i}/{len(x)-H}", end="\r")
        plt.clf()
        plt.plot(x[i:i+H], true_y_steer[i:i+H], label="True", linewidth=1.5)
        plt.plot(x[i:i+H], pred_y_steer[i:i+H], label="Predicted", linewidth=1.5)
        plt.title("Steer")
        plt.ylim([-1.0, 1.0])
        plt.legend()
        plt.savefig("tmp/eval_plot.png", dpi=150)#
        assert os.path.exists("tmp/eval_plot.png")
        images.append(cv2.imread("tmp/eval_plot.png"))
    clip = ImageSequenceClip(images, fps=30)
    clip.write_videofile("eval_plot.mp4")

if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "viz":
        data_viz(*args[1:])
    elif args[0] == "test":
        test_model(*args[1:])
    else:
        raise ValueError("Invalid command")