# 步骤一：特征提取
准备好 COCO 数据集 (如 train2017 目录与 captions_train2017.json)，并在脚本中配置好相对路径。
python 1_extract_base_features.py


# 步骤二：模型微调与特征增强
读取上一步生成的原始特征，训练残差适配器，生成最终特征库。
python 2_train_adapter.py


# 步骤三：实验评估 (可选)
运行自动化测试脚本，输出精度对比表格与系统性能数据。
python 3_evaluate_metrics.py


# 步骤四：启动 Web 检索终端
确保当前目录下存在 finetuned_features.pt 和图片目录，启动交互式界面。
streamlit run app.py
(默认体验账号 - 用户名: admin / 密码: 123456)

# 准备工作
代码是在wsl上面跑的，python环境通过minniconda下载
CLIP_main是从官方地址直接下载的
