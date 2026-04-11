import os, torch, pandas as pd, numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# 配置
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
DATASET_NAMES = ["Brain", "Liver", "Kidney", "HEK293T", "HeLa", "HepG2", "A549", "HEK293", "HCT116", "MOLM13", "CD8T"]
MAX_LEN = 101 # 序列长度
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = "extracted_features"

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

def find_file_ignore_case(target_name):
    for f in os.listdir('.'):
        if f.lower() == target_name.lower(): return f
    return None

def extract_and_save(tk, model, file_path, save_prefix):
    if not file_path or not os.path.exists(file_path): return
    df = pd.read_csv(file_path, sep='\t')
    s_col = [c for c in df.columns if c.lower() in ['text', 'sequence', 'seq']][0]
    l_col = [c for c in df.columns if c.lower() in ['label', 'target', 'y']][0]
    
    all_s, all_l = [], []
    for _, row in df.iterrows():
        seq = str(row[s_col]).upper().replace('U', 'T')
        mid = len(seq) // 2
        # 提取中心 101bp
        processed_s = seq[mid-50 : mid+51] if len(seq) >= MAX_LEN else seq.ljust(MAX_LEN, 'N')
        all_s.append(processed_s)
        all_l.append(int(float(row[l_col])))

    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_s), BATCH_SIZE), desc=f"Extracting {save_prefix}"):
            batch_texts = all_s[i : i + BATCH_SIZE]
            inputs = tk(batch_texts, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            # 提取第 20 层特征
            layer_feat = outputs.hidden_states[20].cpu().numpy().astype(np.float16) 
            features.append(layer_feat)

    np.save(f"{SAVE_DIR}/{save_prefix}_x.npy", np.concatenate(features))
    np.save(f"{SAVE_DIR}/{save_prefix}_y.npy", np.array(all_l))

def main():
    tk = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    
    for ds_name in DATASET_NAMES:
        print(f"\nProcessing Tissue: {ds_name}")
        train_f = find_file_ignore_case(f"{ds_name}_train.tsv")
        test_f = find_file_ignore_case(f"{ds_name}_test.tsv")
        
        extract_and_save(tk, model, train_f, f"{ds_name}_train")
        extract_and_save(tk, model, test_f, f"{ds_name}_test")

if __name__ == "__main__":
    main()