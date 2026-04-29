import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cpu')
model = CNN()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

def predict(image):
    if image is None:
        return "请上传一张图片", None
    
    img = Image.fromarray(image.astype('uint8'), mode='RGB')
    img = img.convert('L')
    img = img.resize((28, 28))
    img_np = np.array(img) / 255.0
    img_np = 1 - img_np
    
    x = torch.tensor(img_np, dtype=torch.float32).flatten().unsqueeze(0)
    
    with torch.no_grad():
        output = model(x)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()
    
    top3 = torch.topk(prob, 3)
    top3_vals = top3.values[0].numpy()
    top3_indices = top3.indices[0].numpy()
    
    top3_text = "\n".join([f"第{i+1}名: 数字{top3_indices[i]} (置信度: {top3_vals[i]*100:.1f}%)" for i in range(3)])
    
    fig, ax = plt.subplots()
    ax.bar(range(10), prob[0].numpy())
    ax.set_xticks(range(10))
    ax.set_xlabel('数字')
    ax.set_ylabel('概率')
    ax.set_title('预测概率分布')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return f"预测结果: {pred}\n\n{top3_text}", buf

if __name__ == "__main__":
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="numpy"),
        outputs=[gr.Textbox(label="预测结果"), gr.Image(type="pil")],
        title="MNIST 手写数字识别",
        description="上传手写数字图片进行识别"
    )
    demo.launch()