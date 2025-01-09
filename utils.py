from moviepy import ImageSequenceClip
import os
import pandas as pd
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import random
from functools import lru_cache
import numpy as np
from pathlib import Path

import sys, time

from models import *

# plt.switch_backend('Agg')

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

        val_x = torch.Tensor(len(val_indices), *self.IMG_SIZE)
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
    
class DKData(Data):
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
        ])

        self.crop_cv2 = lambda x: x[40:, :, :] # HWC. 40px from top

        self.load_from_disk = load
        self.path = path
        self.IMG_SIZE = (3, 80, 160)

    def __getitem__(self, idx_any):
        idx = int(idx_any)
        image_path = os.path.join(self.path, "images", self.data.iloc[idx]["cam/image_array"])
        if self.load_from_disk:
            image = cv2.imread(image_path)
            image = self.crop_cv2(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = image_path
        # throttle, steer
        actuation = torch.tensor([self.data.iloc[idx]["user/throttle"], self.data.iloc[idx]["user/angle"]], dtype=torch.float32)

        return image, actuation
    
class CarlaData(Data):
    def __init__(self, path="data", load=True):
        # .
        # |---data
        # |--->|---images
        # |--->|---data_log.csv

        data = pd.read_csv(os.path.join(path, "data_log.csv"))
        self.data = data

        _len = len(data)
        print(f"Loaded {len(data)} samples")

        # filter out rows where the image doesn't exist
        def filter_fn(row):
            img_paths = (row["img_path"], row["img_path_l"], row["img_path_r"])
            for img_path in img_paths:
                path = str(Path(img_path).resolve())
                if not os.path.exists(path):
                    return False
            return True
        
        self.data = self.data[self.data.apply(filter_fn, axis=1)] # filter out rows where the image doesn't exist
        print(f"Filtered out {_len - len(self.data)} samples")
        
        # image is 3 800x600 side by side. resize to (3*224, 168)
        self.transform = transforms.Compose([
            transforms.Resize((168, 3*224)),
            transforms.ToTensor(),
        ])

        self.load_from_disk = load
        self.IMG_SIZE = (3, 168, 3*224)

        self.headers = [
            "img_path_l","img_path","img_path_r","throttle","steer",
            "way1_x","way1_y","way2_x","way2_y","way3_x","way3_y","pos_x","pos_y", "turntype"
        ]
    
    @lru_cache() # cache so we don't have to read from disk every time
    def __getitem__(self, idx_any):
        idx = int(idx_any)
        data = {}
        for header in self.headers:
            data[header] = self.data.iloc[idx][header]
        if self.load_from_disk:
            paths = [data["img_path_l"], data["img_path"], data["img_path_r"]] # 3 paths
            paths = [str(Path(path).resolve()) for path in paths]
            try:
                images = [cv2.imread(path) for path in paths] # read images
                images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images] # cv2 reads them as BGR
            except Exception as e:
                # remove the image from the data frame
                self.data.drop(idx, inplace=True)
                # print warning (image path)
                print(f"Error reading image at {paths}")
                # raise an exception so we can skip this frame
                raise e
            concat_img = np.concatenate(images, axis=1) # axes HWC (168, 3*224, 3)
            image = Image.fromarray(concat_img)
            image = self.transform(image)
        else:
            image = data["img_path"]

        # output model prediction will be
        # [ steer    w1x w2x w3x]
        # [ throttle w1y w2y w3y]
        predict_data = torch.empty((2, 4))
        predict_data[0] = torch.tensor([data["steer"], data["way1_x"], data["way2_x"], data["way3_x"]])
        predict_data[1] = torch.tensor([data["throttle"], data["way1_y"], data["way2_y"], data["way3_y"]])

        flat_preds = predict_data.cpu().detach().numpy().flatten(order="F") # (str, thr, w1x, w1y, w2x, w2y, w3x, w3y)
        flat_preds = torch.Tensor(flat_preds)
        return image, flat_preds

    def balanced_indices(self):
        # Get turntype indices
        idx_straight = self.data.index[self.data["turntype"] == "straight"].tolist()
        idx_left = self.data.index[self.data["turntype"] == "left"].tolist()
        idx_right = self.data.index[self.data["turntype"] == "right"].tolist()

        # Find largest count
        max_count = max(len(idx_straight), len(idx_left), len(idx_right))

        # Oversample smaller sets
        def oversample(index_list):
            if len(index_list) == 0:
                return []
            reps = max_count // len(index_list)
            remainder = max_count % len(index_list)
            return index_list * reps + random.sample(index_list, remainder)

        balanced = oversample(idx_straight) + oversample(idx_left) + oversample(idx_right)
        random.shuffle(balanced)
        return balanced
    
    def index_generator(self):
        indices = self.balanced_indices()
        for idx in indices:
            yield idx

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
    model = MegaPilotNet(drop=0.0)
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

    BATCH_SIZE = 16
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
        images = torch.stack(images).to("cuda")
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

@torch.no_grad()
def test_waypoint_model(model_pth, data_pth):
    model = MultiCamWaypointNet(drop=0.0)
    model.load_state_dict(torch.load(model_pth))
    model = model.to("cuda")
    model.eval()
    data = CarlaData(data_pth)

    true_y = []
    pred_y = []
    x = []

    # to choose subset of data
    data.data = data.data[2560:4096]

    BATCH_SIZE = 1024
    chunk_ends = [i for i in range(0, len(data), BATCH_SIZE)]
    chunk_ends.append(len(data))

    chunks = [i for i in zip(chunk_ends[:-1], chunk_ends[1:])]

    idx_buckets = [list(range(a, b)) for a, b in chunks]

    t0 = time.perf_counter()
    for bucket in tqdm(idx_buckets, desc="Inference"):
        images = []
        ground_truths = []
        for i in bucket:
            image, ground_truth = data[i]
            images.append(image)
            ground_truths.append(ground_truth)
        images = torch.stack(images).to("cuda")
        ground_truths = torch.stack(ground_truths)
        # old code: actually inference
        out = model.forward(images)
        for i, out in enumerate(out):
            pred_y.append(out)
            true_y.append(ground_truths[i])
            x.append(i + bucket[0])
    t1 = time.perf_counter()
    print(f"Inference Time per frame: {(t1-t0)/len(data)}")

    plt.cla()
    input("Enter to start viz")

    # subplots.
    '''
    | imageL | image | imageR | waypoints_pred/true |
    | steer_pred/true-------- | waypoints_pred/true |
    | throttle_pred/true----- | waypoints_pred/true |
    '''
    plt.ion()
    fig_cam, axs_cam = plt.subplots(1, 2, figsize=(12, 5))
    img_plot = axs_cam[0].imshow(np.zeros((600, 3*800, 3), dtype=np.uint8))
    axs_cam[0].axis('off')
    ax_way = axs_cam[1]
    
    for i in range(len(x)):
        image = data[i][0]
        pred_vals = pred_y[i].cpu().detach().numpy()
        gt_vals = true_y[i]

        # Update camera image subplot
        concat_img = (image.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        img_plot.set_data(concat_img)
        
        # Clear and plot top-down waypoints
        ax_way.clear()
        ax_way.axis('equal')
        ax_way.set_xbound(-50, 50)
        ax_way.set_ybound(-50, 50)
        ax_way.plot(0, 0, 'bo', label='Vehicle')
        # parse predicted w1, w2, w3
        ax_way.plot(
            [pred_vals[3], pred_vals[5], pred_vals[7]],
            [pred_vals[2], pred_vals[4], pred_vals[6]], 'r-x', label='Predicted',
            alpha=0.75
        )
        # parse ground-truth w1, w2, w3
        ax_way.plot(
            [gt_vals[3], gt_vals[5], gt_vals[7]],
            [gt_vals[2], gt_vals[4], gt_vals[6]], 'g-x', label='Ground Truth',
            alpha=0.75
        )
        ax_way.legend(loc='upper right')
        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    args = sys.argv[1:]
    if args[-1] and args[-1] == "dk":
        dk = True
    if args[0] == "viz":
        data_viz(*args[1:-1], dk)
    elif args[0] == "test":
        test_model(*args[1:-1], dk)
    elif args[0] == "waypoint":
        test_waypoint_model(*args[1:])
    else:
        raise ValueError("Invalid command")