import torch
import ujson
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv
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

def create_graph_from_puzzle(input_grid, output_grid):
    padded = [row for row in input_grid] + [[0]*5]
    nodes = []
    for i in range(5):
        row_feat = padded[i]
        for j in range(5):
            nodes.append(row_feat)

    x = torch.tensor(nodes, dtype=torch.float)  # shape: (25,5)

    edge_index = []
    grid_size = 5
    for i in range(grid_size):
        for j in range(grid_size):
            curr = i*grid_size + j
            for k in range(grid_size):
                if k != j:
                    nbr = i*grid_size + k
                    edge_index.append([curr, nbr])
            for k in range(grid_size):
                if k != i:
                    nbr = k*grid_size + j
                    edge_index.append([curr, nbr])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # (2, E)

    y = torch.tensor(output_grid, dtype=torch.long).view(-1)  # (25,)
    return Data(x=x, edge_index=edge_index, y=y)

consts, sols = load_data()
dataset = [create_graph_from_puzzle(c, s) for c, s in zip(consts, sols)]

n_total = len(dataset)
n_train = int(0.95 * n_total)
indices = list(range(n_total))
random.shuffle(indices)
train_dataset = [dataset[i] for i in indices[:n_train]]
val_dataset   = [dataset[i] for i in indices[n_train:]]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def total_same_row_and_column_occurrences_5x5(grid):
    total = 0
    for i in range(5):
        seen_row = set()
        seen_col = set()
        for j in range(5):
            vr = grid[i][j]
            vc = grid[j][i]
            if vr in seen_row:
                total += 1
            else:
                seen_row.add(vr)
            if vc in seen_col:
                total += 1
            else:
                seen_col.add(vc)
    return total

class CustomPuzzleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, preds, targets):
        # preds: (N,1), targets: (N,)
        preds_flat = preds.view(-1)
        loss1 = self.mae(preds_flat, targets.float())

        batch_size = targets.size(0) // 25
        preds_np = preds_flat.view(batch_size, 25).detach().cpu().numpy()
        total_dup = 0
        for grid_flat in preds_np:
            grid = grid_flat.reshape(5,5)
            total_dup += total_same_row_and_column_occurrences_5x5(grid)
        loss2 = (total_dup / batch_size) * 0.1
        out_of_bounds = ((preds_flat < 0.5) | (preds_flat > 5.5)).any().item()
        if(out_of_bounds):
            return loss1 + loss2 + 1000
        return loss1 + loss2

class PuzzleGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(5, 16, heads=4, concat=False)
        self.conv2 = GATConv(16, 64, heads=4, concat=False)   
        self.conv3 = GCNConv(64, 64)
        self.conv4 = GATConv(64, 128, heads=4, concat=False)
        self.conv5 = GATConv(128, 64, heads=4, concat=False)
        self.conv6 = GCNConv(64, 64)
        self.conv7 = GATConv(64, 32, heads=4, concat=False)
        self.conv8 = GCNConv(32, 32)
        self.conv9 = GCNConv(32, 16)
        self.dropout = nn.Dropout(0.2)
        self.mlp = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = F.relu(self.conv6(x, edge_index))
        x = F.relu(self.conv7(x, edge_index))
        x = F.relu(self.conv8(x, edge_index))
        x = F.relu(self.conv9(x, edge_index))
        x = self.mlp(x)
        x = torch.round(x.float())
        return x

model = PuzzleGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = CustomPuzzleLoss()
model.load_state_dict(torch.load("puzzle_gnn_model.pth"))
# for epoch in tqdm(range(5)):
#     model.train()
#     total_loss = 0
#     for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
#         batch = batch.to(device)
#         optimizer.zero_grad()
#         out = model(batch.x, batch.edge_index)  # (N,1)
#         loss = criterion(out, batch.y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")
# torch.save(model.state_dict(), "puzzle_gnn_model.pth") # 15 MAE 5 MSE


model.eval()
results = []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        results.append([out, batch.y])

print(results[0])
