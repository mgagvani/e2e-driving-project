import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
    torch.set_default_device("cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", torch.cuda.get_device_name(), " with properties: ", torch.cuda.get_device_properties(device))
    
    from utils import Data, DKData
    from models import PilotNet, ResNet18Pilot

    args = sys.argv[1:]
    if len(args) == 1:
        args = [args[0], None]
    print("Args: ", args)
    if len(args) == 0:
        data = Data()
    elif args[0] == "dk":
        if args[1] and args == "resnet":
            print("Loading DKData with resnet")
            data = DKData(resnet=True)
        else:
            data = DKData()

    BATCH_SIZE = 1

    # get_tensors must be replaced with DataLoader to prevent OOM
    train_dataset, val_dataset = data.train_val_split()
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device="cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device="cuda")
    )

    data_filter = lambda x: abs(x[1]) > 0.2
    use_filter = False

    if args[1] == "resnet":
        print("Using ResNet18Pilot")
        model = ResNet18Pilot().to(device)
    else:
        model = PilotNet().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    losses, skipped_vals = [], []
    best_loss = float("inf")
    for epoch in range(20):
        for i, (train_x, train_y) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", colour="green", total=len(train_loader)):
            (train_x, train_y) = (train_x.to(device), train_y.to(device))
            # NOTE: Data filter is skipped
            optimizer.zero_grad()
            out = model(train_x)
            loss = criterion(out, train_y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            avg_loss = 0
            for i, (val_x, val_y) in tqdm(enumerate(val_loader), desc=f"Epoch {epoch} Validation", colour="red", total=len(val_loader)):
                (val_x, val_y) = (val_x.to(device), val_y.to(device))
                out = model(val_x)
                avg_loss += criterion(out, val_y)
            losses.append(avg_loss.item() / len(val_loader))
            if losses[-1] < best_loss:
                best_loss = losses[-1]
                print(f"Saving model with loss {best_loss}")
                torch.save(model.state_dict(), "model.pth")

        print(f"Epoch {epoch}, Loss: {losses[-1]}")

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

