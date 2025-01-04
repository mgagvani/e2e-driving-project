import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

def weighted_mse_loss(y, y_hat):
    '''
    y: tensor of shape (batch_size, 2)
    y_hat: tensor of shape (batch_size, 2)

    returns: weighted mean squared error loss

    Basically, weight MSE by steering value. 
    We want to penalize large steering errors.
    '''
    return y[:, 0] * torch.mean((y - y_hat) ** 2)

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

    # data_filter = lambda x: 0.01 < abs(x[1]) < 1.5
    data_filter = lambda x: sum(abs(x)) > 3. or abs(x[0]) < 0.05
    use_filter = True


    # configure GPU
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", torch.cuda.get_device_name())

    model = PilotNet()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = weighted_mse_loss

    losses, skipped_vals = [], []
    best_loss = float("inf")
    for epoch in range(20):
        print() # NOTE DEBUG
        avg_train_loss, train_data = 0, []
        for i in range(len(train_x)):
            # filter out unwanted data
            if data_filter(train_y[i]) and use_filter: # filter out according to data_filter
                skipped_vals.append(train_y[i][1])
                continue
            # coin flip --> horizontal flip
            if torch.rand(1).item() > 0.5:
                train_x[i] = train_x[i].flip(2)
                train_y[i] = torch.tensor([-train_y[i][0], train_y[i][1]])
            if i % 1000 == 0:
                print(f"epoch {epoch} train {i}/{len(train_x)}", end="\r")
            optimizer.zero_grad()
            out = model(train_x[i].unsqueeze(0))
            # print(f"out: {out}, train_y: {train_y[i]}", end='\r') # NOTE DEBUG
            loss = criterion(out, train_y[i].unsqueeze(0))
            avg_train_loss += (_loss:=loss.item())
            train_data.append((list(out.detach().cpu().numpy()), list(train_y[i].detach().cpu().numpy()), _loss, i))
            # print(f"loss: {loss.item()}", end='\r') # NOTE DEBUG
            loss.backward()
            optimizer.step()

        print("Epoch", epoch, "Train Loss:", avg_train_loss / len(train_x)) # NOTE DEBUG
        # sort by top 10 losses
        sorted_train_losses = sorted(train_data, key=lambda x: x[2], reverse=True)
        print("Top 10 losses:")
        for data in sorted_train_losses[:10]:
            print(data)

        with torch.no_grad():
            avg_loss = 0
            for i in range(len(val_x)):
                if i % 1000 == 0:
                    print(f"epoch {epoch} val {i}/{len(val_x)}", end="\r")
                out = model(val_x[i].unsqueeze(0))
                avg_loss += criterion(out, val_y[i].unsqueeze(0))
            losses.append(avg_loss.item() / len(val_x))
            if losses[-1] < best_loss:
                best_loss = loss
                print(f"Saving model with loss {loss.item()}")
                torch.save(model.state_dict(), "model.pth")

        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # skip values
    print(f"Skipped {len(skipped_vals)} values")
    print(f"Mean skipped value: {sum(skipped_vals) / len(skipped_vals)}")
    print(f"Min/max/median skipped value: {min(skipped_vals), max(skipped_vals), sorted(skipped_vals)[len(skipped_vals) // 2]}")

    # plot losses
    # TODO fix bug
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss.png")

