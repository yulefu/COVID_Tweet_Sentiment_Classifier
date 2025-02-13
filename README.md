# COVID Tweet Sentiment Classifier

## Table of Contents
- [Introduction](#introduction)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Inference](#inference)
- [Exporting to ONNX](#exporting-to-onnx)
- [Extending and Improving the Model](#extending-and-improving-the-model)
- [中文说明](#中文说明)

---
## Introduction

This project is a text classification model that determines whether a given Twitter tweet during the COVID-19 pandemic is **positive**, **moderately positive**, **neutral**, **moderately negative**, or **negative**. In my previous experiments, I tried older models such as Lexicon-based methods, TF-IDF, and even the conventional use of BERT. For this project, I wanted a challenge—so I decided to try a deep learning model based on a feedforward neural network (also known as a fully connected network).

To start, I use BERT, but not in the typical fine-tuning way. Instead, **BERT is used as a feature extractor**. My model consists of **three main components**:

1. **BERT Model (`self.bert`)**  
   - This is the **pretrained** `bert-base-uncased` model from Hugging Face.
   - It processes the input text and produces a **768-dimensional feature vector** for each input sentence.
   - This layer is used primarily for feature extraction (it is not fully trainable unless you decide to fine-tune).

2. **Feature Processing Layer (`self.feature_layer`)**  
   - A **fully connected layer** that combines the BERT output (768 dimensions) with 3 additional handcrafted features (resulting in 771 dimensions in total).
   - This layer includes:
     - **`nn.Linear(768 + 3, 512)`** → A linear transformation that reduces the dimensionality to 512.
     - **ReLU activation (`nn.ReLU()`)** → Introduces non-linearity.
     - **Dropout (`nn.Dropout(0.3)`)** → Helps prevent overfitting.

3. **Classification Layer (`self.classifier`)**  
   - A final **fully connected layer** that maps the 512 features to 5 output classes (corresponding to the five sentiment categories).
   - Implemented as **`nn.Linear(512, num_classes)`**, where `num_classes` equals 5.

For training, I use **CrossEntropyLoss**, which computes the difference between the model's predicted probability distribution and the actual labels. Mathematically, it is given by:

$$\text{Loss} = - \sum_{i} y_i \log(\hat{y}_i)$$

where:  
- \( y_i \) is the true label, and  
- \( \hat{y}_i \) is the predicted probability for class \( i \).

Finally, the model is trained using gradient descent via the **AdamW** optimizer. This optimizer computes the gradient of the loss and updates the model’s weights in the direction that minimizes the loss.

---

## Overview

The COVID Tweet Sentiment Classifier leverages a pre-trained BERT model combined with a fully connected network to determine the sentiment of tweets during the COVID-19 pandemic. It classifies tweets into one of five sentiment categories ranging from extremely negative to extremely positive.

Key points:
- **BERT Backbone:** Utilizes the `bert-base-uncased` model for robust language feature extraction.
- **Feature Augmentation:** Combines the BERT-derived 768-dimensional feature with 3 handcrafted features.
- **Feedforward Neural Network:** Processes the combined features through a fully connected network to produce a final prediction.
- **Training & Loss:** Uses CrossEntropyLoss with the AdamW optimizer.

---

## Project Structure

```
COVID_Tweet_Sentiment_Classifier/
│
├── data/
│   ├── Corona_NLP_train.csv      # Training dataset CSV
│   └── Corona_NLP_test.csv       # Testing dataset CSV
│
├── models/
│   ├── covid_model.pth           # Saved PyTorch model checkpoint
│   └── covid_model.onnx          # Exported ONNX model file (optional)
│
├── src/
│   ├── model.py                  # Contains the COVIDTweetClassifier class
│   ├── dataset.py                # Contains the TweetDataset class and data loading function
│   ├── train.py                  # Training loop and early stopping logic
│   ├── inference.py              # Inference classes for PyTorch and ONNX pipelines
│   └── utils.py                  # Utility functions (logging configuration, feature extraction)
│
├── README.md                     # This README file
└── requirements.txt              # Python dependencies
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/COVID_Tweet_Sentiment_Classifier.git
   cd COVID_Tweet_Sentiment_Classifier
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows use: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Data Preparation

Place your CSV files (`Corona_NLP_train.csv` and `Corona_NLP_test.csv`) in the `data/` folder. The `load_data` function in `src/dataset.py`:
- Reads the CSV files using pandas.
- Cleans the tweet text and maps sentiment labels (e.g., “Extremely Negative” to 0).
- Extracts three handcrafted features from each tweet.
- Returns a list of tweet texts, a tensor of features, and a list of labels.

---

## Training the Model

To train the model, run:

```bash
python src/train.py
```

This script:
- Loads and processes the data.
- Initializes the BERT-based classifier.
- Trains the model for a set number of epochs (default is 3 for runtime optimization).
- Evaluates on the test set after each epoch using the weighted F1 score.
- Saves the best model checkpoint to `models/covid_model.pth`.

---

## Inference

Two inference pipelines are provided:

### PyTorch Inference

The `TweetPredictor` class in `src/inference.py`:
- Loads the saved PyTorch model.
- Preprocesses new tweet text (tokenization and feature extraction).
- Returns the sentiment label with the highest predicted score.

Example:

```python
from src.inference import TweetPredictor

predictor = TweetPredictor(model_path='models/covid_model.pth')
print(predictor.predict("Vaccine distribution is going great!"))
```

### ONNX Inference

The `ONNXPredictor` class in `src/inference.py`:
- Loads the ONNX model using ONNX Runtime.
- Processes the input tweet text similarly.
- Outputs the sentiment label.

Example:

```python
from src.inference import ONNXPredictor

onnx_predictor = ONNXPredictor(model_path='models/covid_model.onnx')
print(onnx_predictor.predict("Vaccine distribution is going great!"))
```

---

## Exporting to ONNX

After training, you can export the PyTorch model to ONNX format with:

```python
# Uncomment and run after training:
dummy_input = (
    torch.randint(0, 10000, (1, 128)),    # Simulated token ids
    torch.ones(1, 128),                  # Attention mask
    torch.randn(1, 3)                    # Handcrafted features
)
torch.onnx.export(model, dummy_input, "models/covid_model.onnx")
```

---

## Extending and Improving the Model

To further improve the current ~85% accuracy or build additional functionalities, consider:
- **Using Domain-Specific Pre-trained Models:**  
  Experiment with models like [BERTweet](https://github.com/VinAIResearch/BERTweet) or [COVID-Twitter-BERT](https://github.com/cdqa-suite/covid-twitter-bert).
- **Enhancing Handcrafted Features:**  
  Add more features (e.g., sentiment lexicon scores, emoji interpretations, user metadata).
- **Multi-Task Learning:**  
  Extend the model to perform additional tasks (e.g., topic detection, sarcasm recognition).
- **Deployment:**  
  Create a web app using Flask or FastAPI for real-time sentiment analysis.
- **Data Augmentation:**  
  Use techniques like back-translation or synonym replacement to enrich the training dataset.


---
## Credits

Special thanks to the Hugging Face Transformers and PyTorch communities for their excellent tools, as well as to all researchers whose work contributed to the ideas and methodologies used in this project.

---

## 中文说明

## 目录

- [概述](#概述)
- [项目结构](#项目结构)
- [安装](#安装)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [推理](#推理)
- [导出 ONNX 模型](#导出-onnx-模型)
- [模型扩展与改进](#模型扩展与改进)


### 引言

本项目是一个文本分类模型，用于判断 COVID-19 疫情期间的 Twitter 推文情感类别：分别为**积极**、**中度积极**、**中立**、**中度消极**和**消极**。之前我尝试过许多旧模型，如词典法、TF-IDF 以及传统的 BERT 方法。对于本项目，我希望接受一个挑战，因此决定尝试一种基于深度学习的全连接神经网络模型。

在本项目中，我使用 BERT 但并非以传统方式微调，而是作为特征提取器。我的模型由**三个主要部分**组成：

1. **BERT 模型 (`self.bert`)**  
   - 这是 Hugging Face 提供的预训练 `bert-base-uncased` 模型。  
   - 它处理输入文本并为每个句子生成一个**768 维特征向量**。  
   - 除非进行微调，否则该层主要用于特征提取，并非完全可训练。

2. **特征处理层 (`self.feature_layer`)**  
   - 这是一个全连接层，将 BERT 输出（768 维）与 3 个手工提取的特征合并（总维度为 771）。  
   - 层结构包括：  
     - **`nn.Linear(768 + 3, 512)`** → 将维度降至 512 的全连接层。  
     - **ReLU 激活函数 (`nn.ReLU()`)** → 增加非线性。  
     - **Dropout 层 (`nn.Dropout(0.3)`)** → 防止过拟合。

3. **分类层 (`self.classifier`)**  
   - 最后一个全连接层，将 512 维特征映射到 5 个输出类别（对应 5 种情感）。  
   - 通过 **`nn.Linear(512, num_classes)`** 实现，其中 `num_classes` 为 5。

在训练时，我使用 **CrossEntropyLoss** 来计算模型预测概率分布与实际标签之间的差异，其数学表达式为：

$$\text{Loss} = - \sum_{i} y_i \log(\hat{y}_i)$$

其中：  
- \( y_i \) 为真实标签；  
- \( \hat{y}_i \) 为类别 \( i \) 的预测概率。

最后，通过 **AdamW 优化器** 使用梯度下降法训练模型。该优化器通过计算损失的梯度并更新模型权重以最小化损失。

---

## 概述

COVID 推文情感分类器使用 PyTorch 和 Hugging Face Transformers 库构建。模型利用预训练的 `bert-base-uncased` 模型从 [CLS] 标记中提取 768 维向量，再拼接三个手工提取的特征（归一化词数、标点强度、URL 出现标志），形成一个扩展的表示。随后，该表示经过一个全连接网络进行处理，最终输出五个情感类别的 logits。

关键点：
- **BERT 主干：** 使用 `bert-base-uncased` 模型提供强大的语言理解能力。
- **手工特征：** 增加了三个简单的数值特征，捕获推文特有的信息。
- **数据管道：** 使用自定义的 PyTorch `Dataset` 和 `DataLoader` 类。
- **训练循环：** 包含基于加权 F1 分数的早停机制。
- **推理：** 提供了 PyTorch 和 ONNX 两种推理方式，便于部署。

---

## 项目结构

```
COVID_Tweet_Sentiment_Classifier/
│
├── data/
│   ├── Corona_NLP_train.csv      # 训练数据 CSV 文件
│   └── Corona_NLP_test.csv       # 测试数据 CSV 文件
│
├── models/
│   ├── covid_model.pth           # 保存的 PyTorch 模型检查点
│   └── covid_model.onnx          # 导出的 ONNX 模型（可选）
│
├── src/
│   ├── model.py                  # 定义 COVIDTweetClassifier 类
│   ├── dataset.py                # 定义 TweetDataset 类和数据加载函数
│   ├── train.py                  # 训练循环和早停逻辑
│   ├── inference.py              # 推理类（包括 PyTorch 和 ONNX 推理）
│   └── utils.py                  # 工具函数（日志配置、特征提取等）
│
├── README.md                     # 本说明文件
└── requirements.txt              # Python 依赖包列表
```

---

## 安装

1. **克隆仓库：**

   ```bash
   git clone https://github.com/your-username/COVID_Tweet_Sentiment_Classifier.git
   cd COVID_Tweet_Sentiment_Classifier
   ```

2. **创建并激活虚拟环境（可选，但推荐）：**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows 下使用: venv\Scripts\activate
   ```

3. **安装依赖包：**

   ```bash
   pip install -r requirements.txt
   ```

---

## 数据准备

将 CSV 文件（`Corona_NLP_train.csv` 和 `Corona_NLP_test.csv`）放置于 `data/` 目录中。`src/dataset.py` 中的 `load_data` 函数将：
- 使用 pandas 读取 CSV 文件；
- 清洗推文文本，并将情感标签（如“Extremely Negative”映射为 0）转换为数字；
- 为每条推文提取三个手工特征；
- 返回推文列表、特征张量以及标签列表。

---

## 模型训练

运行训练脚本：

```bash
python src/train.py
```

此脚本将：
- 加载并预处理数据；
- 初始化基于 BERT 的模型和附加的特征层；
- 在固定的训练轮数内（默认 3 个 epoch 用于演示优化运行时间）训练模型；
- 在每个 epoch 后使用加权 F1 分数对测试集进行评估；
- 将最佳模型（基于 F1 分数提升）保存到 `models/covid_model.pth`。

日志系统配置为显示带时间戳的训练进度更新。

---

## 推理

提供两种推理方式：

### PyTorch 推理

在 `src/inference.py` 中的 `TweetPredictor` 类：
- 加载保存的 PyTorch 模型；
- 对新推文进行预处理（包括分词和特征提取）；
- 根据最高 logits 输出情感标签。

示例代码：

```python
from src.inference import TweetPredictor

predictor = TweetPredictor(model_path='models/covid_model.pth')
print(predictor.predict("Vaccine distribution is going great!"))
```

### ONNX 推理

在 `src/inference.py` 中的 `ONNXPredictor` 类：
- 使用 ONNX Runtime 加载 ONNX 模型；
- 以类似方式处理文本；
- 输出预测情感标签（使用 NumPy 数组）。

示例代码：

```python
from src.inference import ONNXPredictor

onnx_predictor = ONNXPredictor(model_path='models/covid_model.onnx')
print(onnx_predictor.predict("Vaccine distribution is going great!"))
```

---

## 导出 ONNX 模型

训练完成后，你可以将 PyTorch 模型导出为 ONNX 格式以便在其他平台上进行快速推理。训练脚本中提供了以下注释代码片段：

```python
# 导出到 ONNX (训练后运行)
dummy_input = (
    torch.randint(0, 10000, (1, 128)),    # 模拟 token ids
    torch.ones(1, 128),                  # 注意力 mask
    torch.randn(1, 3)                    # 手工特征
)
torch.onnx.export(model, dummy_input, "models/covid_model.onnx")
```

取消注释并运行此代码即可生成 ONNX 文件。

---

## 模型扩展与改进

目前模型准确率约为 85%。你可以通过以下方向进一步改进模型：
- **更好的预训练模型：**  
  尝试使用专门针对推文预训练的模型，如 [BERTweet](https://github.com/VinAIResearch/BERTweet) 或 [COVID-Twitter-BERT](https://github.com/cdqa-suite/covid-twitter-bert)。
- **附加特征：**  
  增加更多手工特征（例如情感词典得分、表情符号解释、用户元数据）。
- **多任务学习：**  
  扩展模型以同时执行多个任务（如主题检测、讽刺识别）。
- **应用开发：**  
  构建基于 Flask 或 FastAPI 的 Web 应用，实现实时情感分析。
- **数据增强：**  
  利用反向翻译或同义词替换扩充训练数据。

---

## Credits

特别感谢 Hugging Face Transformers 和 PyTorch 社区的所有贡献者，以及所有为此项目提供数据和文献支持的研究者们。

