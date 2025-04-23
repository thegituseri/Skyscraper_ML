import torch
import ujson
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
seed = 42
random.seed(seed)
torch.manual_seed(seed)

def load_data(constraint_file="constraints_5x5.json", solution_file="solutions_5x5.json"):
    with open(constraint_file, "r") as f:
        constraints_list = ujson.load(f)
    with open(solution_file, "r") as f:
        solutions_list = ujson.load(f)
    return constraints_list, solutions_list

puzzle, solutions = load_data()
class PuzzleDataset(Dataset):
    def __init__(self, puzzle, solutions):
        self.consts, self.sols = puzzle, solutions

    def __len__(self):
        return len(self.consts)

    def __getitem__(self, idx):
        inp = torch.tensor(self.consts[idx], dtype=torch.float)      # (4,5)
        pad = torch.zeros(1, 5, dtype=torch.float)                  # (1,5)
        grid5x5 = torch.cat([inp, pad], dim=0)                      # (5,5)
        grid5x5 = grid5x5.unsqueeze(0)                             # (1,4,5)
        target = torch.tensor(self.sols[idx], dtype=torch.float)    # (5,5)
        return grid5x5, target

def total_same_row_and_column_occurrences_5x5(grid):
    total = 0
    for i in range(5):
        seen_r, seen_c = set(), set()
        for j in range(5):
            vr, vc = grid[i][j], grid[j][i]
            if vr in seen_r: total += 1
            else: seen_r.add(vr)
            if vc in seen_c: total += 1
            else: seen_c.add(vc)
    return total

class CustomPuzzleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.MSELoss()

    def forward(self, preds, targets):
        loss1 = self.mae(preds, targets)
        preds_np = preds.detach().cpu().numpy()
        dup = sum(total_same_row_and_column_occurrences_5x5(g) for g in preds_np)
        penalty = (dup / preds.size(0)) * 0.1
        return loss1 + penalty

class PuzzleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_block1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.proj_up1 = nn.Conv2d(1, 8, kernel_size=1)

        self.up_block2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=(5,1), padding=(2,0)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8,16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.proj_up2 = nn.Conv2d(8, 16, kernel_size=1)

        self.up_block3 = nn.Sequential(
            nn.Conv2d(16,16, kernel_size=(5,1), padding=(2,0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.proj_up3 = nn.Conv2d(16, 32, kernel_size=1)

        self.down_block1 = nn.Sequential(
            nn.ConvTranspose2d(32,16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16,16, kernel_size=(5,1), padding=(2,0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16,16, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.proj_down1 = nn.Conv2d(32,16, kernel_size=1)

        self.down_block2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=(5,1), padding=(2,0)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.proj_down2 = nn.Conv2d(16, 8, kernel_size=1)

        self.down_block3 = nn.Sequential(
            nn.ConvTranspose2d(8, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=3, padding=1),
        )
        self.proj_down3 = nn.Conv2d(8, 1, kernel_size=1)


    def forward(self, x):
        u1 = self.up_block1(x)
        r1 = F.relu(u1 + self.proj_up1(x))

        u2 = self.up_block2(r1)
        r2 = F.relu(u2 + self.proj_up2(r1))

        u3 = self.up_block3(r2)
        r3 = F.relu(u3 + self.proj_up3(r2))

        d1 = self.down_block1(r3)
        s1 = F.relu(d1 + self.proj_down1(r3))

        d2 = self.down_block2(s1)
        s2 = F.relu(d2 + self.proj_down2(s1))

        d3 = self.down_block3(s2)
        out = torch.round(d3)

        return out.view(-1, 5, 5)

dataset = PuzzleDataset(puzzle, solutions)
train_size = int(0.95 * len(dataset))
val_size   = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PuzzleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = CustomPuzzleLoss()
model.load_state_dict(torch.load("Model1.pth"))
if 1:
    for epoch in range(30):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), "Model1.pth")

model.eval()
arr = []
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        arr.append((preds, y))
print(arr[0])
