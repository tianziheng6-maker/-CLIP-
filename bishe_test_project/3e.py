import torch
import faiss
import numpy as np
import os

# ================= ⚙️ 路径配置 =================
TEST_RAW_FILE = "./test_raw_features.pt"
ADAPTER_WEIGHTS = "./finetuned_coco_adapter.pt"

class ResidualAdapter(torch.nn.Module):
    def __init__(self, feature_dim=512):
        super(ResidualAdapter, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, feature_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(feature_dim * 2, feature_dim)
        )
    def forward(self, x):
        return x + self.mlp(x)

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
        
        # ---- 已彻底删除原本存在于此处的 random 干扰代码 ----

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
    print("正在执行全量指标评估（真实数据）...")

    if not os.path.exists(TEST_RAW_FILE):
        print(f"❌ 错误: 找不到测试特征文件，请先运行 1_extract_base_features.py 提取 val2017")
        return
    
    test_data = torch.load(TEST_RAW_FILE, map_location="cpu")
    img_raw = test_data["image_features"]
    txt_raw = test_data["text_features"]
    img_names = np.array(test_data["image_filenames"])
    txt_info = test_data["text_info"]

    # 1. 计算 Baseline (CLIP)
    baseline_res = calculate_metrics(img_raw, txt_raw, img_names, txt_info)

    # 2. 计算 Ours (MLP)
    adapter = ResidualAdapter().cpu()
    if os.path.exists(ADAPTER_WEIGHTS):
        adapter.load_state_dict(torch.load(ADAPTER_WEIGHTS, map_location="cpu"))
    adapter.eval()
    
    with torch.no_grad():
        img_fine = adapter(img_raw.float())
        # 注意：检索前通常需要对特征进行归一化以保证余弦相似度准确
        img_fine /= img_fine.norm(dim=-1, keepdim=True)
    
    ours_res = calculate_metrics(img_fine, txt_raw, img_names, txt_info)

    # 3. 输出论文标准对比表
    print("\n" + "="*90)
    print(f"{'模型方案':<21} | {'R@1 (%)':<10} | {'R@5 (%)':<10} | {'R@10 (%)':<10} | {'MRR':<10}")
    print("-" * 90)
    fmt = "{:<25} | {:<10.2f} | {:<10.2f} | {:<10.2f} | {:<10.4f}"
    print(fmt.format("Baseline (Original CLIP)", baseline_res['R@1'], baseline_res['R@5'], baseline_res['R@10'], baseline_res['MRR']))
    print(fmt.format("Ours (CLIP + MLP ) ", ours_res['R@1'], ours_res['R@5'], ours_res['R@10'], ours_res['MRR']))
    print("="*90)

if __name__ == "__main__":
    main()