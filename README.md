# MNIST 手写数字识别实验

基于 PyTorch 的 CNN 手写数字识别实验项目。

## 👤 学生信息

- **姓名**：马艺杰
- **学号**：112311170336
- **班级**：数据1231

## 📁 项目结构
├── app_flask.py # Web应用（支持手写输入） ├── model.pth # 训练好的CNN模型 ├── simple_save_model.py # 模型训练脚本 ├── test_model.py # 模型测试脚本 ├── train_experiments.py # 对比实验脚本 ├── requirements.txt # 依赖列表 ├── CNN手写数字识别实验模板.md # 实验报告 └── digit-recognizer (1)/ # Kaggle数据集
## 🚀 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 启动Web应用
python app_flask.py

# 访问地址
# http://127.0.0.1:5000
```

## 📊 实验结果

- **Kaggle Score**: 0.9876
- **模型结构**: CNN (2层卷积 + 2层全连接)
- **优化器**: Adam

## 🛠️ 技术栈

- Python 3.12
- PyTorch 2.3.1
- Flask (Web部署)
- PIL (图像处理)
