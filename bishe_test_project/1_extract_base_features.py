import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm

# ================= 配置路径 (WSL 内部路径) =================
# 建议确保图片和 JSON 标注文件已放在当前工作目录下或指定绝对路径
# IMAGE_DIR = "./train2017" 
IMAGE_DIR = "./val2017/val2017" 
# ANNOTATION_FILE = "./annotations/captions_train2017.json"
ANNOTATION_FILE = "./annotations/captions_val2017.json"

OUTPUT_FILE = "./test_raw_features.pt"
BATCH_SIZE = 128  # 根据你的显存大小调整，显存小请改为 64 或 32

def extract_features():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 使用设备: {device}")

    # 1. 加载 OpenAI 原始 CLIP 模型 (不参与训练，仅作为特征提取器)
    #
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # 2. 解析 COCO 标注文件
    if not os.path.exists(ANNOTATION_FILE):
        print(f"❌ 找不到标注文件: {ANNOTATION_FILE}")
        return

    with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 建立 Image_ID 到文件名映射
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    # 提取图文对数据
    dataset = []
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in id_to_filename:
            dataset.append({
                "filename": id_to_filename[img_id],
                "caption": ann['caption']
            })

    total_data = len(dataset)
    print(f"📊 准备从 {total_data} 条 COCO 数据中提取特征...")

    # --- 调试代码开始 ---
    if total_data == 0:
        print(f"❌ 错误: dataset 为空，请检查 JSON 标注文件内容或 Image_ID 映射。")
    else:
        # 抽样检查前三张图片的物理路径是否存在
        for check_item in dataset[:3]:
            check_path = os.path.join(IMAGE_DIR, check_item["filename"])
            if not os.path.exists(check_path):
                print(f"⚠️ 警告: 找不到图片路径 -> {os.path.abspath(check_path)}")
    # --- 调试代码结束 ---

    all_image_features = []
    all_text_features = []
    saved_filenames = []
    saved_captions = []

    # 3. 执行特征提取过程
    with torch.no_grad():
        for i in tqdm(range(0, total_data, BATCH_SIZE), desc="⏳ 提取中"):
            batch_data = dataset[i:i + BATCH_SIZE]

            images = []
            texts = []
            valid_batch_filenames = []
            valid_batch_captions = []

            for item in batch_data:
                img_path = os.path.join(IMAGE_DIR, item["filename"])
                if not os.path.exists(img_path):
                    continue

                try:
                    # 图像预处理
                    img = Image.open(img_path).convert("RGB")
                    images.append(preprocess(img))
                    # 文本分词与截断
                    texts.append(clip.tokenize(item["caption"], truncate=True)[0])

                    valid_batch_filenames.append(item["filename"])
                    valid_batch_captions.append((item["filename"], item["caption"]))
                except Exception:
                    continue

            if not images:
                continue

            # 转换为 Tensor 并送入设备
            image_input = torch.stack(images).to(device)
            text_input = torch.stack(texts).to(device)

            # 核心：计算特征向量并进行 L2 归一化
            img_feat = model.encode_image(image_input)
            txt_feat = model.encode_text(text_input)

            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

            # 转存至 CPU 内存，释放 GPU 空间
            all_image_features.append(img_feat.cpu())
            all_text_features.append(txt_feat.cpu())
            saved_filenames.extend(valid_batch_filenames)
            saved_captions.extend(valid_batch_captions)

    # 4. 汇总并落盘保存
    torch.save({
        "image_features": torch.cat(all_image_features, dim=0),
        "text_features": torch.cat(all_text_features, dim=0),
        "image_filenames": saved_filenames,
        "text_info": saved_captions
    }, OUTPUT_FILE)

    print(f"✅ 原始特征已保存至 {OUTPUT_FILE}，准备进入下一步对比实验！")

if __name__ == "__main__":
    extract_features()