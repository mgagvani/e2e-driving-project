import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
plt.switch_backend("Agg")
    
   
def main_PilotNet():
    # configure GPU
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", torch.cuda.get_device_name(), " with properties: ", torch.cuda.get_device_properties(device))

    from utils import Data
    from models import PilotNet, SigLIPPilot
    data = Data()

    train_x, train_y, val_x, val_y = data.get_tensors()

    data_filter = lambda x: abs(x[1]) > 0.2
    use_filter = False

    model = PilotNet().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    losses, skipped_vals = [], []
    best_loss = float("inf")
    print()
    for epoch in range(20):
        for i in range(len(train_x)):
            if data_filter(train_y[i]) and use_filter: # filter out according to data_filter
                skipped_vals.append(train_y[i][1])
                continue
            if i % 1000 == 0:
                print(f"epoch {epoch} train {i}/{len(train_x)}", end="\r")
            optimizer.zero_grad()
            out = model(train_x[i].unsqueeze(0))
            loss = criterion(out, train_y[i].unsqueeze(0))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            avg_loss = 0
            for i in range(len(val_x)):
                if i % 1000 == 0:
                    print(f"epoch {epoch} val {i}/{len(val_x)}", end="\r")
                out = model(val_x[i].unsqueeze(0))
                avg_loss += criterion(out, val_y[i].unsqueeze(0))
            losses.append(avg_loss.item() / len(val_x))
            if losses[-1] < best_loss:
                best_loss = losses[-1]
                print(f"Saving model with loss {loss.item()}")
                torch.save(model.state_dict(), "model.pth")

        print(f"Epoch {epoch}, Loss: {avg_loss.item()}")

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

