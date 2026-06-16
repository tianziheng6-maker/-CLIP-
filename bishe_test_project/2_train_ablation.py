import torch
import torch.nn as nn
import torch.optim as optim
import os

# ================= 🔧 消融配置区 (运行三次，每次改这里) =================
# MODE = "full"      # 第一次运行：完整方案 -> adapter_full.pt
MODE = "no_res"    # 第二次运行：无残差 -> adapter_no_res.pt
# MODE = "no_drop"     # 第三次运行：无 Dropout -> adapter_no_drop.pt

RAW_FEATURES_PATH = "./raw_clip_features.pt" # 你的训练特征
SAVE_PATH = f"./adapter_{MODE}.pt"
EPOCHS = 20
BATCH_SIZE = 512
LR = 1e-4
# =====================================================================

class AblationAdapter(nn.Module):
    def __init__(self, feature_dim=512, mode="full"):
        super(AblationAdapter, self).__init__()
        self.mode = mode
        layers = [nn.Linear(feature_dim, feature_dim * 2), nn.GELU()]
        if mode != "no_drop":
            layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(feature_dim * 2, feature_dim))
        self.mlp = nn.Sequential(*layers)
        
        if mode != "no_res":
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        if self.mode != "no_res":
            return x + self.mlp(x)
        else:
            return self.mlp(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在训练模式: {MODE} | 设备: {device}")

    # 1. 加载特征数据
    if not os.path.exists(RAW_FEATURES_PATH):
        print(f"❌ 找不到特征文件: {RAW_FEATURES_PATH}")
        return
    
    data = torch.load(RAW_FEATURES_PATH, map_location=device)
    img_feats = data["image_features"].float()
    txt_feats = data["text_features"].float()

    # 2. 初始化模型、损失函数和优化器
    model = AblationAdapter(mode=MODE).to(device)
    criterion = nn.CosineEmbeddingLoss() # 让图文向量余弦相似度更高
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3. 训练循环
    model.train()
    target = torch.ones(BATCH_SIZE).to(device) # 1 表示让它们靠近

    for epoch in range(EPOCHS):
        running_loss = 0.0
        # 简单分批训练
        for i in range(0, len(img_feats), BATCH_SIZE):
            img_batch = img_feats[i:i+BATCH_SIZE].to(device)
            txt_batch = txt_feats[i:i+BATCH_SIZE].to(device)
            
            if len(img_batch) < BATCH_SIZE: continue # 跳过最后一个不满的 batch

            optimizer.zero_grad()
            # 通过适配器修正图像特征
            outputs = model(img_batch)
            # 计算修正后的图像与对应文本的损失
            loss = criterion(outputs, txt_batch, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(img_feats):.6f}")

    # 4. 保存结果
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ 权重已保存至: {SAVE_PATH}\n")

if __name__ == "__main__":
    train()