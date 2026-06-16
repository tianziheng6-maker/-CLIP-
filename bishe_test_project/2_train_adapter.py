import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# ================= ⚙️ 配置路径 =================
INPUT_FEATURE_FILE = "./raw_clip_features.pt"
ADAPTER_WEIGHTS = "./finetuned_coco_adapter.pt"
FINAL_FEATURES_FILE = "./finetuned_features.pt" # 修正后的特征库，供app使用

# ================= 1. 定义残差适配器 (MLP) =================
class ResidualAdapter(nn.Module):
    def __init__(self, feature_dim=512):
        super(ResidualAdapter, self).__init__()
        # 设计一个 bottleneck 结构：512 -> 1024 -> 512
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        # 关键初始化策略：将最后一层权重初始化为0
        # 这样模型在训练开始时输出几乎为原向量，保证训练的稳定性（残差初值为0）
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        # 残差连接：y = x + MLP(x)
        return x + self.mlp(x)

def train_adapter():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 训练启动 | 设备: {device}")

    # 2. 加载原材料（1_extract_base_features.py 生成的文件）
    if not os.path.exists(INPUT_FEATURE_FILE):
        print(f"❌ 找不到特征文件: {INPUT_FEATURE_FILE}")
        return

    data = torch.load(INPUT_FEATURE_FILE)
    img_feats = data["image_features"].float()
    txt_feats = data["text_features"].float()
    
    # 构建 DataLoader
    dataset = TensorDataset(img_feats, txt_feats)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 3. 实例化模型与优化器
    model = ResidualAdapter(feature_dim=512).to(device)
    
    # 使用余弦嵌入损失：目标是让匹配的对相似度趋近于 1
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 4. 训练循环
    epochs = 10 # 毕设建议 10-20 epoch 即可，防止在 COCO 上过拟合
    model.train()
    
    print("🧪 开始微调特征空间...")
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_img, batch_txt in dataloader:
            batch_img, batch_txt = batch_img.to(device), batch_txt.to(device)
            
            # y=1 表示这两者应该是匹配的
            target = torch.ones(batch_img.size(0)).to(device)

            optimizer.zero_grad()
            output_img = model(batch_img)
            
            # 这里的逻辑是：通过优化图片端特征，使其更靠近对应的文本特征
            loss = criterion(output_img, batch_txt, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # 5. 保存 MLP 权重
    torch.save(model.state_dict(), ADAPTER_WEIGHTS)
    print(f"✅ 适配器权重已保存: {ADAPTER_WEIGHTS}")

    # 6. 生成“修正后”的特征库用于后续检索系统 (app.py)
    print("✨ 正在生成最终特征库...")
    model.eval()
    with torch.no_grad():
        # 对全量图像特征进行修正
        fine_tuned_img_feats = model(img_feats.to(device)).cpu()
        # 重新归一化（非常重要：MLP输出后必须重新L2归一化，否则检索精度会崩）
        fine_tuned_img_feats /= fine_tuned_img_feats.norm(dim=-1, keepdim=True)
        
    torch.save({
        "image_features": fine_tuned_img_feats,
        "text_features": txt_feats, # 文本端保持原始 CLIP 特征
        "image_filenames": data["image_filenames"],
        "text_info": data["text_info"]
    }, FINAL_FEATURES_FILE)
    print(f"🎉 最终特征库已生成: {FINAL_FEATURES_FILE}")

if __name__ == "__main__":
    train_adapter()