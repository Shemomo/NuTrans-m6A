import os, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from model import NuTrans_m6A
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_NAMES = ["Brain", "Liver", "Kidney", "HEK293T", "HeLa", "HepG2", "A549", "HEK293", "HCT116", "MOLM13", "CD8T"]
BASE_DIR = "extracted_features"
CHECKPOINTS = [f"nutrans_best_fold{i}.pth" for i in range(1, 6)]
OUTPUT_CSV = "nutrans_m6a_ensemble_results.csv"

class MmapTestDataset(Dataset):
    def __init__(self, dataset_names, base_dir=BASE_DIR):
        self.x_mmap_list, self.y_mmap_list, self.cumulative_sizes = [], [], [0]
        self.total_size = 0
        for name in dataset_names:
            x_p, y_p = f"{base_dir}/{name}_test_x.npy", f"{base_dir}/{name}_test_y.npy"
            if os.path.exists(x_p):
                self.x_mmap_list.append(np.load(x_p, mmap_mode='r'))
                self.y_mmap_list.append(np.load(y_p, mmap_mode='r'))
                self.total_size += len(self.y_mmap_list[-1]); self.cumulative_sizes.append(self.total_size)

    def __len__(self): return self.total_size
    def __getitem__(self, i):
        ds_idx = np.searchsorted(self.cumulative_sizes, i, side='right') - 1
        local_idx = i - self.cumulative_sizes[ds_idx]
        return torch.from_numpy(np.array(self.x_mmap_list[ds_idx][local_idx])).to(torch.float32), \
               torch.tensor(self.y_mmap_list[ds_idx][local_idx]).to(torch.float32)

def main():
    test_ds = MmapTestDataset(DATASET_NAMES)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    y_true = []
    for _, by in loader: y_true.extend(by.numpy())
    y_true = np.array(y_true)
    
    ensemble_probs = []
    for ckpt in CHECKPOINTS:
        print(f" Loading {ckpt} for inference...")
        model = NuTrans_m6A().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        
        probs = []
        with torch.no_grad():
            for bx, _ in loader:
                p = torch.sigmoid(model(bx.to(DEVICE))).cpu().numpy().flatten()
                probs.append(p)
        ensemble_probs.append(np.concatenate(probs))
    
    avg_probs = np.mean(ensemble_probs, axis=0)
    pd.DataFrame({'true_label': y_true, 'ensemble_prob': avg_probs}).to_csv(OUTPUT_CSV, index=False)
    print(f"完成！结果已保存至 {OUTPUT_CSV}")
    
    # 最终汇总
    y_pred = (avg_probs > 0.5).astype(int)
    print(f"\n NuTrans-m6A Final Performance:")
    print(f"AUC: {roc_auc_score(y_true, avg_probs):.4f}")
    print(f"MCC: {matthews_corrcoef(y_true, y_pred):.4f}")

if __name__ == "__main__":
    main()