import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
plt.switch_backend("Agg")
class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.nn6 = nn.Linear(1152, 100)
        self.nn7 = nn.Linear(100, 50)
        self.nn8 = nn.Linear(50, 10)
        self.nn9 = nn.Linear(10, 2) # throttle, steer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.nn6(x))
        x = F.relu(self.nn7(x))
        x = F.relu(self.nn8(x))
        x = self.nn9(x)

        return x
    
   
if __name__ == "__main__":
    from utils import Data
    data = Data()

    train_x, train_y, val_x, val_y = data.get_tensors()

    # data_filter = lambda x: abs(x[1]) > 0.2
    data_filter = lambda x: True # no filter, basically

    # configure GPU
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", torch.cuda.get_device_name())

    model = PilotNet()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    losses, skipped_vals = [], []
    best_loss = float("inf")
    for epoch in range(20):
        for i in range(len(train_x)):
            if data_filter(train_y[i]): # filter out according to data_filter
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
            for i in range(len(val_x)):
                if i % 1000 == 0:
                    print(f"epoch {epoch} val {i}/{len(val_x)}", end="\r")
                out = model(val_x[i].unsqueeze(0))
                loss = criterion(out, val_y[i].unsqueeze(0))
            losses.append(loss.item())
            if loss < best_loss:
                best_loss = loss
                print(f"Saving model with loss {loss.item()}")
                torch.save(model.state_dict(), "model.pth")

        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # skip values
    print(f"Skipped {len(skipped_vals)} values")
    print(f"Mean skipped value: {sum(skipped_vals) / len(skipped_vals)}")
    print(f"Min/max/median skipped value: {min(skipped_vals), max(skipped_vals), sorted(skipped_vals)[len(skipped_vals) // 2]}")

    # plot losses
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss.png")

