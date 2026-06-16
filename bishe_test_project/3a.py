import torch
import faiss
import numpy as np
import os

# ================= ⚙️ 路径配置 =================
TEST_RAW_FILE = "./test_raw_features.pt"
ADAPTER_WEIGHTS = "./finetuned_coco_adapter.pt"

# 消融权重路径配置
ABLATION_WEIGHTS = {
    "Ablation (Full)": "./adapter_full.pt",
    "Ablation (No Residual)": "./adapter_no_res.pt",
    "Ablation (No Dropout)": "./adapter_no_drop.pt"
}

class ResidualAdapter(torch.nn.Module):
    def __init__(self, feature_dim=512, mode="full"):
        super(ResidualAdapter, self).__init__()
        self.mode = mode
        layers = [
            torch.nn.Linear(feature_dim, feature_dim * 2),
            torch.nn.GELU()
        ]
        # 根据模式动态决定是否添加 Dropout
        if mode != "no_drop":
            layers.append(torch.nn.Dropout(0.2))
        layers.append(torch.nn.Linear(feature_dim * 2, feature_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        # 根据模式决定是否使用残差连接
        if self.mode != "no_res":
            return x + self.mlp(x)
        else:
            return self.mlp(x)

def calculate_metrics(img_feats, txt_feats, img_names, txt_info):
    """
    计算 Recall@K 和 MRR 指标
    """
    img_feats = img_feats.float().numpy().astype('float32')
    txt_feats = txt_feats.float().numpy().astype('float32')
    
    d = img_feats.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(img_feats)

    k_max = 10
    D, I = index.search(txt_feats, k_max) 

    r1, r5, r10, mrr = 0, 0, 0, 0.0
    num_queries = len(txt_feats)

    for i in range(num_queries):
        target_name = txt_info[i][0]
        retrieved_indices = I[i]
        
        rank = -1
        for r in range(k_max):
            if img_names[retrieved_indices[r]] == target_name:
                rank = r + 1
                break
        

        if rank != -1:
            if rank <= 1: r1 += 1
            if rank <= 5: r5 += 1
            if rank <= 10: r10 += 1
            mrr += 1.0 / rank

    return {
        "R@1": (r1 / num_queries) * 100,
        "R@5": (r5 / num_queries) * 100,
        "R@10": (r10 / num_queries) * 100,
        "MRR": mrr / num_queries
    }

def main():
    print("正在执行全量指标评估 (真实客观数据)...")

    if not os.path.exists(TEST_RAW_FILE):
        print(f"❌ 错误: 找不到测试特征文件")
        return
    
    test_data = torch.load(TEST_RAW_FILE, map_location="cpu", weights_only=False)
    img_raw = test_data["image_features"]
    txt_raw = test_data["text_features"]
    img_names = np.array(test_data["image_filenames"])
    txt_info = test_data["text_info"]

    results = []

    # 1. 计算 Baseline (CLIP)
    baseline_res = calculate_metrics(img_raw, txt_raw, img_names, txt_info)
    results.append(("Baseline (Original CLIP)", baseline_res))

    # 2. 计算 Ours (MLP)
    adapter = ResidualAdapter().cpu()
    if os.path.exists(ADAPTER_WEIGHTS):
        adapter.load_state_dict(torch.load(ADAPTER_WEIGHTS, map_location="cpu", weights_only=False))
        adapter.eval()
        with torch.no_grad():
            img_fine = adapter(img_raw.float())
            img_fine /= img_fine.norm(dim=-1, keepdim=True)
        ours_res = calculate_metrics(img_fine, txt_raw, img_names, txt_info)
        results.append(("Ours (CLIP + MLP ) ", ours_res))

    # 3. 计算三个消融实验变体
    ablation_configs = [
        ("Ablation (Full)", "full"),
        ("Ablation (No Residual)", "no_res"),
        ("Ablation (No Dropout)", "no_drop")
    ]

    for label, mode in ablation_configs:
        w_path = ABLATION_WEIGHTS[label]
        if os.path.exists(w_path):
            model = ResidualAdapter(mode=mode).cpu()
            model.load_state_dict(torch.load(w_path, map_location="cpu", weights_only=False))
            model.eval()
            with torch.no_grad():
                feat_t = model(img_raw.float())
                feat_t /= feat_t.norm(dim=-1, keepdim=True)
                # 统一调用净化后的指标计算函数
                res = calculate_metrics(feat_t, txt_raw, img_names, txt_info)
                results.append((label, res))

    # 输出论文标准对比表
    print("\n" + "="*90)
    print(f"{'模型方案':<21} | {'R@1 (%)':<10} | {'R@5 (%)':<10} | {'R@10 (%)':<10} | {'MRR':<10}")
    print("-" * 90)
    fmt = "{:<25} | {:<10.2f} | {:<10.2f} | {:<10.2f} | {:<10.4f}"
    
    for name, r in results:
        print(fmt.format(name, r['R@1'], r['R@5'], r['R@10'], r['MRR']))
    print("="*90)

if __name__ == "__main__":
    main()