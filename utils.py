from moviepy import ImageSequenceClip
import os
import pandas as pd
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import numpy as np

import sys, time

from models import *

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
        self.IMG_SIZE = (3, 66, 200)

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
        train_x = torch.Tensor(len(train_indices), *self.IMG_SIZE)

        # regression
        # train_y = torch.Tensor(len(train_indices), 2)

        # one hot
        ONEHOT_SIZE = (2, N_BINS)
        train_y = torch.Tensor(len(train_indices), *ONEHOT_SIZE)
        
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

        val_x = torch.Tensor(len(val_indices), *self.IMG_SIZE)
        val_y = torch.Tensor(len(val_indices), *ONEHOT_SIZE)

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
    
class DKData(Data):
    # def __init__(self, path="../mycar/data", load=True:
    def __init__(self, path="/scratch/gilbreth/mgagvani/DK_Dataset/data", load=True):
        # get all *.catalog
        catalogs = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".catalog")]
        data = pd.DataFrame()
        # headers: _index, _session_id, _timestamp_ms, cam/image_array, user/throttle, user/angle, user/mode
        for catalog in catalogs:
            catalog_file = open(catalog, "r")
            for line in catalog_file.readlines():
                _dict = eval(line)
                _dict_df = pd.DataFrame([_dict])
                data = pd.concat([data, _dict_df], ignore_index=True)

        self.data = data

        # crop 40px from top, then to tensor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.unsqueeze(0)),
            torch.nn.Upsample(size=(224, 224), mode="bicubic"),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ## 
        ])

        self.crop_cv2 = lambda x: x[40:, :, :] # HWC. 40px from top

        self.load_from_disk = load
        self.path = path
        # self.IMG_SIZE = (3, 80, 160)
        self.IMG_SIZE = (3, 224, 224)
        self.onehot = True

    def linear_bin(self, x, n_bins=N_BINS):
        """
        Convert continuous value to one-hot array
        n_bins=20 gives 20 possible positions
        """
        # Clip value to [-1, 1]
        x = np.clip(x, -1, 1)
    
        # Create bin edges (-1 to 1 inclusive)
        bins = np.linspace(-1, 1, n_bins+1)
    
        # Get bin index (0 to n_bins-1)
        idx = np.digitize(x, bins) - 1
    
        # Handle edge case for x = 1.0
        idx = np.clip(idx, 0, n_bins-1)

        # Create one-hot array
        arr = np.zeros(n_bins)
        arr[idx] = 1
        return arr

    def idx_to_steer(self, idx, n_bins=N_BINS):
        """Convert bin index back to continuous value"""
        bins = np.linspace(-1, 1, n_bins)
        return bins[idx]

    def __getitem__(self, idx_any):
        idx = int(idx_any)
        image_path = os.path.join(self.path, "images", self.data.iloc[idx]["cam/image_array"])
        if self.load_from_disk:
            image = cv2.imread(image_path)
            # image = self.crop_cv2(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = Image.fromarray(image)
            # print(image.shape, transforms.ToTensor()(image).shape)
            image = self.transform(image)
        else:
            image = image_path
        # throttle, steer
        if self.onehot:
            throttle_arr = self.linear_bin(self.data.iloc[idx]["user/throttle"])
            steer_arr = self.linear_bin(self.data.iloc[idx]["user/angle"])
            actuation = torch.tensor([throttle_arr, steer_arr], dtype=torch.float32)
        else:
            actuation = torch.tensor([self.data.iloc[idx]["user/throttle"], self.data.iloc[idx]["user/angle"]], dtype=torch.float32)

        return image, actuation

def data_viz(save_path="data.mp4", dk=False):
    # load data
    if dk:
        data = DKData(load=False)
    else:
        data = Data(load=False)
    
    images = []
    for i, (image_path, actuation) in enumerate(data): # its only path because we set load=False
        images.append(image_path)

    # create video
    clip = ImageSequenceClip(images, fps=30)
    clip.write_videofile(save_path, codec="libx264")

def test_model(model_pth, data_pth, dk):
    model = ResNet18Pilot()
    model.load_state_dict(torch.load(model_pth))
    model = model.to("cuda")
    model.eval()
    if dk:
        data = DKData(data_pth)
    else:
        data = Data(data_pth)

    true_y_steer = []
    true_y_throttle = []
    pred_y_steer = []
    pred_y_throttle = []
    x = []

    BATCH_SIZE = 2048
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
            images.append(image.squeeze(0))
            actuations.append(actuation)
        images = torch.stack(images).to("cuda")
        actuations = torch.stack(actuations)
        # old code: actually inference
        out = model.forward(images)
        for i, out in enumerate(out):
            # interpret onehot
            out = out.detach().cpu().numpy()
            # print(out)
            throttle_idx, steer_idx = np.argmax(out[0]), np.argmax(out[1])
            # print(throttle_idx, steer_idx)
            out_new = [data.idx_to_steer(steer_idx), data.idx_to_steer(throttle_idx)]
            # print(out)
            # input()
            actuations_new = actuations[i].detach().cpu().numpy()
            steer_actual_idx, throttle_actual_idx = np.argmax(actuations[i][1]), np.argmax(actuations[i][0])
            actuations_new = [data.idx_to_steer(throttle_actual_idx), data.idx_to_steer(steer_actual_idx)]
            pred_y_steer.append(out_new[1])
            pred_y_throttle.append(out_new[0])
            true_y_steer.append(actuations_new[1])
            true_y_throttle.append(actuations_new[0])
            # print(pred_y_steer); input()
            x.append(i + bucket[0])
    t1 = time.perf_counter()
    print(f"Inference Time per frame: {(t1-t0)/len(data)}")

    plt.cla(); plt.clf()
    
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
    HORIZONTAL = H = 250
    if len(x) > 25_000:
        STEP = 50
    else:
        STEP = 10
    for i in range(0, len(x)-H, STEP):
        print(f"Frame {i}/{len(x)-H}", end="\r")
        plt.clf(); plt.cla()
        fig, axs = plt.subplots(2, figsize=(12, 8))
        axs[0].plot(x[i:i+H], true_y_steer[i:i+H], label="True", linewidth=1.5)
        axs[0].plot(x[i:i+H], pred_y_steer[i:i+H], label="Predicted", linewidth=1.5)
        axs[0].set_title("Steer")
        axs[0].set_ylim([-1.0, 1.0])
        axs[0].legend()

        axs[1].plot(x[i:i+H], true_y_throttle[i:i+H], label="True", linewidth=1.5)
        axs[1].plot(x[i:i+H], pred_y_throttle[i:i+H], label="Predicted", linewidth=1.5)
        axs[1].set_title("Throttle")
        axs[1].set_ylim([-1.0, 1.0])
        axs[1].legend()
        plt.savefig("tmp/eval_plot.png", dpi=100)#
        plt.close(fig)
        assert os.path.exists("tmp/eval_plot.png")
        images.append(cv2.imread("tmp/eval_plot.png"))
    clip = ImageSequenceClip(images, fps=30)
    clip.write_videofile("eval_plot.mp4")

if __name__ == "__main__":
    args = sys.argv[1:]
    if args[-1] and args[-1] == "dk":
        dk = True
    if args[0] == "viz":
        data_viz(*args[1:-1], dk)
    elif args[0] == "test":
        test_model(*args[1:-1], dk)
    else:
        raise ValueError("Invalid command")