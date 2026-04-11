import os, torch, torch.nn as nn, numpy as np, random, json, pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from torch import amp
from model import NuTrans_m6A  # 引用模型名

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128 
EPOCHS = 12
DATASET_NAMES = ["Brain", "Liver", "Kidney", "HEK293T", "HeLa", "HepG2", "A549", "HEK293", "HCT116", "MOLM13", "CD8T"]
LOG_CSV = "nutrans_m6a_training_log.csv"

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
seed_everything(42)

class MmapTrainDataset(Dataset):
    def __init__(self, dataset_names, base_dir="extracted_features"):
        self.x_mmap_list, self.y_mmap_list = [], []
        self.cumulative_sizes = [0]
        self.total_size = 0
        all_labels_list = []
        for name in dataset_names:
            x_p, y_p = f"{base_dir}/{name}_train_x.npy", f"{base_dir}/{name}_train_y.npy"
            if not os.path.exists(x_p): continue
            self.x_mmap_list.append(np.load(x_p, mmap_mode='r'))
            y_data = np.load(y_p, mmap_mode='r')
            self.y_mmap_list.append(y_data)
            all_labels_list.append(np.load(y_p)) # 预读标签用于分层
            self.total_size += len(y_data)
            self.cumulative_sizes.append(self.total_size)
        self.all_labels = np.concatenate(all_labels_list)

    def __len__(self): return self.total_size
    def __getitem__(self, i):
        ds_idx = np.searchsorted(self.cumulative_sizes, i, side='right') - 1
        local_idx = i - self.cumulative_sizes[ds_idx]
        x = torch.from_numpy(np.array(self.x_mmap_list[ds_idx][local_idx])).to(torch.float32)
        y = torch.tensor(self.y_mmap_list[ds_idx][local_idx]).to(torch.float32)
        return x, y

def main():
    train_dataset = MmapTrainDataset(DATASET_NAMES)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(train_dataset)), train_dataset.all_labels)):
        print(f"\n NuTrans-m6A Training | Fold {fold+1}/5")
        model = NuTrans_m6A().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scaler = amp.GradScaler()

        train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(Subset(train_dataset, val_idx), batch_size=BATCH_SIZE, num_workers=4)

        best_auc = 0
        for epoch in range(EPOCHS):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                with amp.autocast(device_type='cuda'):
                    out = model(bx.to(DEVICE))
                    loss = criterion(out, by.to(DEVICE).unsqueeze(1))
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

            model.eval()
            y_true, y_prob = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    p = torch.sigmoid(model(bx.to(DEVICE)))
                    y_prob.extend(p.cpu().numpy().flatten()); y_true.extend(by.numpy())
            
            auc = roc_auc_score(y_true, y_prob)
            print(f"  Ep {epoch+1} | AUC: {auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), f"nutrans_best_fold{fold+1}.pth")
        
        del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()