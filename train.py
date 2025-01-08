import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import sys

import matplotlib.pyplot as plt
plt.switch_backend("Agg")
    
   
def main_PilotNet():
    # configure GPU
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", torch.cuda.get_device_name(), " with properties: ", torch.cuda.get_device_properties(device))
    
    from utils import Data, DKData, CarlaData
    from models import MultiCamWaypointNet

    args = sys.argv[1:]
    if len(args) == 0:
        data = Data()
    elif args[0] == "dk":
        data = DKData()
    elif args[0] == "carla":
        data = CarlaData()

    indices = data.balanced_indices()
    train_idx = indices[:int(len(indices) * 0.8)]
    val_idx = indices[int(len(indices) * 0.8):]

    print(f"Training on {len(train_idx)} samples, validating on {len(val_idx)} samples")

    data_filter = lambda x: abs(x[1]) > 0.2
    use_filter = False

    model = MultiCamWaypointNet().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    losses, train_losses, skipped_vals = [], [], []
    best_loss = float("inf")
    for epoch in range(20):
        avg_train_loss = 0
        for i, idx in enumerate(train_idx):
            # if data_filter(train_y[i]) and use_filter: # filter out according to data_filter
            #     skipped_vals.append(train_y[i][1])
            #     continue
            if i % 1000 == 0:
                print(f"epoch {epoch} train {i}/{len(train_idx)}", end="\r")
            optimizer.zero_grad()
            curr_x, curr_y = data[idx]
            out = model(curr_x.unsqueeze(0))
            loss = criterion(out, curr_y[i].unsqueeze(0))
            avg_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_losses.append(avg_train_loss / len(train_idx))

        with torch.no_grad():
            avg_loss = 0
            for i, idx in enumerate(val_idx):
                if i % 1000 == 0:
                    print(f"epoch {epoch} val {i}/{len(val_idx)}", end="\r")
                val_x, val_y = data[idx]
                out = model(val_x.unsqueeze(0))
                avg_loss += criterion(out, val_y.unsqueeze(0))
            losses.append(avg_loss.item() / len(val_idx))
            if losses[-1] < best_loss:
                best_loss = losses[-1]
                print(f"Saving model with VAL loss {best_loss}")
                torch.save(model.state_dict(), "model.pth")

        print(f"Epoch {epoch}, TRAIN loss: {train_losses[-1]}, VAL loss: {losses[-1]}")

    # skip values
    try:
        print(f"Skipped {len(skipped_vals)} values")
        print(f"Mean skipped value: {sum(skipped_vals) / len(skipped_vals)}")
        print(f"Min/max/median skipped value: {min(skipped_vals), max(skipped_vals), sorted(skipped_vals)[len(skipped_vals) // 2]}")
    except ZeroDivisionError:
        print("No skipped values")
    except Exception as e:
        print(f"Error in skipped values: {e}")

    # plot losses
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss.png")

if __name__ == "__main__":
    main_PilotNet()

