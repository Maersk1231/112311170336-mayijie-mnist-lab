from flask import Flask, request, Response
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import base64
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

app = Flask(__name__)

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 MNIST 手写数字识别</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
            max-width: 900px;
            width: 100%;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            align-items: start;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .canvas-section {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .canvas-wrapper {
            background: linear-gradient(145deg, #f8f9fa, #e9ecef);
            border-radius: 16px;
            padding: 15px;
            box-shadow: 
                inset 0 2px 10px rgba(0, 0, 0, 0.1),
                0 4px 20px rgba(0, 0, 0, 0.1);
        }
        
        #canvas {
            border-radius: 12px;
            background: white;
            cursor: crosshair;
            box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 12px;
        }
        
        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 50px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }
        
        .btn-secondary {
            background: #f1f3f4;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #e8eaed;
            transform: translateY(-2px);
        }
        
        .result-section {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .result-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 30px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }
        
        .result-card .prediction {
            font-size: 6rem;
            font-weight: bold;
            text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 10px;
        }
        
        .result-card .label {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .top3-card {
            background: #f8f9fa;
            border-radius: 16px;
            padding: 20px;
        }
        
        .top3-card h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .top3-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .top3-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease;
        }
        
        .top3-item:hover {
            transform: translateX(5px);
        }
        
        .top3-item .rank {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.85rem;
        }
        
        .top3-item .digit {
            font-size: 1.5rem;
            font-weight: bold;
            color: #333;
            margin: 0 15px;
        }
        
        .top3-item .prob-bar-wrapper {
            flex: 1;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .top3-item .prob-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .top3-item .prob-value {
            margin-left: 12px;
            font-weight: 600;
            color: #667eea;
            min-width: 50px;
            text-align: right;
        }
        
        .prob-dist-card {
            background: #f8f9fa;
            border-radius: 16px;
            padding: 20px;
        }
        
        .prob-dist-card h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .prob-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
        }
        
        .prob-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }
        
        .prob-item .digit-circle {
            width: 45px;
            height: 45px;
            border-radius: 50%;
            background: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            font-weight: bold;
            color: #333;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        
        .prob-item:hover .digit-circle {
            transform: scale(1.1);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .prob-item .bar-container {
            width: 100%;
            height: 60px;
            background: #e9ecef;
            border-radius: 8px;
            position: relative;
            display: flex;
            align-items: flex-end;
            overflow: hidden;
        }
        
        .prob-item .bar-fill {
            width: 100%;
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px 8px 0 0;
            transition: height 0.5s ease;
            position: relative;
        }
        
        .prob-item .bar-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        }
        
        .prob-item .prob-text {
            font-size: 0.85rem;
            font-weight: 600;
            color: #666;
        }
        
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #999;
            font-size: 0.9rem;
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .pulse {
            animation: pulse 1s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 MNIST 手写数字识别</h1>
            <p>在画板上写下数字，体验 AI 的识别能力</p>
        </div>
        
        <div class="main-content">
            <div class="canvas-section">
                <div class="canvas-wrapper">
                    <canvas id="canvas" width="280" height="280"></canvas>
                </div>
                <div class="controls">
                    <button class="btn btn-primary" onclick="recognize()">
                        <span>🔍</span>
                        <span>识别</span>
                    </button>
                    <button class="btn btn-secondary" onclick="clearCanvas()">
                        <span>🗑️</span>
                        <span>清空</span>
                    </button>
                </div>
            </div>
            
            <div class="result-section">
                <div id="result-card" class="result-card" style="display: none;">
                    <div class="prediction" id="prediction">?</div>
                    <div class="label">预测数字</div>
                </div>
                
                <div id="top3-card" class="top3-card" style="display: none;">
                    <h3>🏆 Top-3 预测</h3>
                    <div class="top3-list" id="top3-list"></div>
                </div>
                
                <div id="prob-dist-card" class="prob-dist-card" style="display: none;">
                    <h3>📊 概率分布</h3>
                    <div class="prob-grid" id="prob-grid"></div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>基于 PyTorch + CNN 实现 | MNIST 数据集</p>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const resultCard = document.getElementById('result-card');
        const predictionDiv = document.getElementById('prediction');
        const top3Card = document.getElementById('top3-card');
        const top3List = document.getElementById('top3-list');
        const probDistCard = document.getElementById('prob-dist-card');
        const probGrid = document.getElementById('prob-grid');
        
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#1a1a1a';
        ctx.lineWidth = 18;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;
        
        canvas.addEventListener('mousedown', (e) => {
            isDrawing = true;
            [lastX, lastY] = getPosition(e);
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            const [x, y] = getPosition(e);
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();
            [lastX, lastY] = [x, y];
        });
        
        canvas.addEventListener('mouseup', () => {
            isDrawing = false;
        });
        
        canvas.addEventListener('mouseout', () => {
            isDrawing = false;
        });
        
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            isDrawing = true;
            const touch = e.touches[0];
            [lastX, lastY] = getPosition(touch);
        });
        
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (!isDrawing) return;
            const touch = e.touches[0];
            const [x, y] = getPosition(touch);
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();
            [lastX, lastY] = [x, y];
        });
        
        canvas.addEventListener('touchend', () => {
            isDrawing = false;
        });
        
        function getPosition(e) {
            const rect = canvas.getBoundingClientRect();
            return [e.clientX - rect.left, e.clientY - rect.top];
        }
        
        function clearCanvas() {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            resultCard.style.display = 'none';
            top3Card.style.display = 'none';
            probDistCard.style.display = 'none';
        }
        
        async function recognize() {
            const imgData = canvas.toDataURL('image/png');
            
            try {
                resultCard.style.display = 'none';
                top3Card.style.display = 'none';
                probDistCard.style.display = 'none';
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imgData })
                });
                
                const data = await response.json();
                
                predictionDiv.textContent = data.prediction;
                resultCard.classList.add('fade-in');
                resultCard.style.display = 'block';
                
                top3List.innerHTML = '';
                data.top3.forEach((item, index) => {
                    const div = document.createElement('div');
                    div.className = 'top3-item fade-in';
                    div.style.animationDelay = `${index * 0.1}s`;
                    div.innerHTML = `
                        <div class="rank">${index + 1}</div>
                        <div class="digit">${item.digit}</div>
                        <div class="prob-bar-wrapper">
                            <div class="prob-bar" style="width: ${item.prob}%"></div>
                        </div>
                        <div class="prob-value">${item.prob}%</div>
                    `;
                    top3List.appendChild(div);
                });
                top3Card.style.display = 'block';
                
                probGrid.innerHTML = '';
                data.probabilities.forEach((prob, index) => {
                    const div = document.createElement('div');
                    div.className = 'prob-item';
                    div.innerHTML = `
                        <div class="digit-circle">${index}</div>
                        <div class="bar-container">
                            <div class="bar-fill" style="height: ${prob}%"></div>
                        </div>
                        <div class="prob-text">${prob}%</div>
                    `;
                    probGrid.appendChild(div);
                });
                probDistCard.style.display = 'block';
                
            } catch (error) {
                predictionDiv.textContent = '?';
                resultCard.style.display = 'block';
                console.error('识别失败:', error);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_CONTENT

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return Response(json.dumps({'error': 'No image data'}), status=400, mimetype='application/json')
        
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img = img.resize((28, 28))
        img_np = np.array(img) / 255.0
        img_np = 1 - img_np
        
        x = torch.tensor(img_np, dtype=torch.float32).flatten().unsqueeze(0)
        
        with torch.no_grad():
            output = model(x)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
        
        probabilities = [round(float(p) * 100, 1) for p in prob[0].numpy()]
        
        top3_indices = np.argsort(probabilities)[::-1][:3]
        top3 = [{'digit': int(i), 'prob': probabilities[i]} for i in top3_indices]
        
        return Response(json.dumps({
            'prediction': int(pred),
            'probabilities': probabilities,
            'top3': top3
        }), mimetype='application/json')
    
    except Exception as e:
        return Response(json.dumps({'error': str(e)}), status=500, mimetype='application/json')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)