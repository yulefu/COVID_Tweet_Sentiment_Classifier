# COVID Tweet Sentiment Classifier

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
- $y_i$ is the true label, and  
- $\hat{y}_i$ is the predicted probability for class $i$.

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
├── Corona_NLP_train.csv      # Training dataset CSV
└── Corona_NLP_test.csv       # Testing dataset CSV
│
├── models/
│   ├── covid_model.pth           # Saved PyTorch model checkpoint
│   └── covid_model.onnx          # Exported ONNX model file (optional)
│
├── model.py                  # Contains the COVIDTweetClassifier class
├── dataset.py                # Contains the TweetDataset class and data loading function
├── train.py                  # Training loop and early stopping logic
├── predict.py                # Predict classes for PyTorch
├── evaluate.py               # provide accuracy, precision, recall, and F1 score
├── export.py                 # Contains a function for exporting the trained model to the ONNX format
└── onnx_predictor.py         # Predict classes for ONNX
│
├── README.md                     # This README file
└── requirements.txt              # Python dependencies
```

---

## Installation

Make sure the Python verison is 3.10.12. Also, please ensure you have the correct CUDA and cuDNN installed. We are using CUDA version 12.1, and cuDNN Version 90100. 


Download CUDA from NVIDIA’s official website and install it.

Download cuDNN from NVIDIA Developer and install it.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yulefu/COVID_Tweet_Sentiment_Classifier.git
   cd COVID_Tweet_Sentiment_Classifier
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Setup and Usage

### 1. **Data Preparation**
The model expects a dataset in CSV format with the following columns:
- `OriginalTweet`: The tweet text.
- `Sentiment`: The sentiment label for each tweet.

Example of the expected sentiment labels:
- `Extremely Negative`
- `Negative`
- `Neutral`
- `Positive`
- `Extremely Positive`

The datasets are in the `Corona_NLP_train.csv` and `Corona_NLP_test.csv` files for training and testing.

### 2. **Training the Model**

Run the `train.py` script to train the model. The script will load the training data from `Corona_NLP_train.csv`, preprocess it, and train the model using the BERT embeddings along with handcrafted features.

```bash
python train.py
```

During training, the model will save the best performing model to `covid_model.pth` based on the F1 score.

#### Parameters
- **Learning rate**: `2e-5` (can be modified inside `train.py`)
- **Batch size**: `32` for training and `64` for testing
- **Epochs**: `3` (can be adjusted in `train.py`)

### 3. **Model Prediction**

After training, use the `predict.py` file to make predictions on new tweets.

You can use the following code in `predict.py` to load the trained model and make predictions:

```python
from predict import TweetPredictor

predictor = TweetPredictor(model_path='covid_model.pth')
print(predictor.predict("Vaccine distribution is going great!"))
```

#### Example Output:
```
Positive
```

### 4. **Model Evaluation**

To evaluate the model's performance on the test dataset, run the `evaluate.py` file. It will load the test data from `Corona_NLP_test.csv`, and calculate various evaluation metrics including accuracy, precision, recall, and F1 score.

```bash
python evaluate.py
```

### 5. **Export Model to ONNX**

If you want to export the model to the ONNX format for inference with other frameworks, you can use the `export_to_onnx` function in `export.py`. This will export the trained model to `covid_model.onnx`.

```python
from export import export_to_onnx

dummy_input_ids = torch.zeros(1, 128, dtype=torch.int64)
dummy_attention_mask = torch.zeros(1, 128, dtype=torch.int64)
dummy_features = torch.zeros(1, 3, dtype=torch.float32)

export_to_onnx(model, (dummy_input_ids, dummy_attention_mask, dummy_features))
```

### 6. **ONNX Inference**

To make predictions using the exported ONNX model, you can use `ONNXTweetPredictor` in `onnx_predictor.py`. The class uses `onnxruntime` for inference.

```python
from onnx_predictor import ONNXTweetPredictor

onnx_predictor = ONNXTweetPredictor(onnx_model_path='covid_model.onnx')
print(onnx_predictor.predict("Vaccine distribution is going great!"))
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
- $y_i$ 为真实标签；  
- $\hat{y}_i$ 为类别 $i$ 的预测概率。

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
Here’s the Chinese translation of your README:

---

## 项目结构

```
COVID_Tweet_Sentiment_Classifier/
│
├── Corona_NLP_train.csv      # 训练数据集 CSV
└── Corona_NLP_test.csv       # 测试数据集 CSV
│
├── models/
│   ├── covid_model.pth           # 保存的 PyTorch 模型检查点
│   └── covid_model.onnx          # 导出的 ONNX 模型文件（可选）
│
├── model.py                  # 包含 COVIDTweetClassifier 类
├── dataset.py                # 包含 TweetDataset 类和数据加载函数
├── train.py                  # 训练循环和早停逻辑
├── predict.py                # 用于 PyTorch 预测类别
├── evaluate.py               # 提供准确率、精确度、召回率和 F1 分数
├── export.py                 # 包含将训练好的模型导出为 ONNX 格式的函数
└── onnx_predictor.py         # 用于 ONNX 的预测类别
│
├── README.md                     # 本 README 文件
└── requirements.txt              # Python 依赖项
```

---

## 安装

请确保 Python 版本为 3.10.12。此外，请确保您已正确安装 CUDA 和 cuDNN。我们使用的是 CUDA 版本 12.1 和 cuDNN 版本 90100。  

从 NVIDIA 官方网站下载 CUDA 并安装。  

从 NVIDIA Developer 下载 cuDNN 并安装。

1. **克隆仓库：**

   ```bash
   git clone https://github.com/yulefu/COVID_Tweet_Sentiment_Classifier.git
   cd COVID_Tweet_Sentiment_Classifier
   ```

2. **安装所需的包：**

   ```bash
   pip install -r requirements.txt
   ```

---

## 设置和使用

### 1. **数据准备**
模型期望输入的 CSV 数据集应包含以下列：
- `OriginalTweet`: 推文文本。
- `Sentiment`: 每条推文的情感标签。

示例的情感标签：
- `Extremely Negative`（极度负面）
- `Negative`（负面）
- `Neutral`（中立）
- `Positive`（正面）
- `Extremely Positive`（极度正面）

训练和测试数据集分别存储在 `Corona_NLP_train.csv` 和 `Corona_NLP_test.csv` 文件中。

### 2. **训练模型**

运行 `train.py` 脚本来训练模型。该脚本将从 `Corona_NLP_train.csv` 加载训练数据，对其进行预处理，并使用 BERT 嵌入和手工特征训练模型。

```bash
python train.py
```

在训练过程中，模型会根据 F1 分数保存最佳性能的模型到 `covid_model.pth`。

#### 参数
- **学习率**：`2e-5`（可以在 `train.py` 中修改）
- **批量大小**：训练时为 `32`，测试时为 `64`
- **轮次**：`3`（可以在 `train.py` 中调整）

### 3. **模型预测**

训练完成后，可以使用 `predict.py` 文件对新的推文进行预测。

可以在 `predict.py` 中使用以下代码加载训练好的模型并进行预测：

```python
from predict import TweetPredictor

predictor = TweetPredictor(model_path='covid_model.pth')
print(predictor.predict("Vaccine distribution is going great!"))
```

#### 示例输出：
```
Positive
```

### 4. **模型评估**

要评估模型在测试数据集上的表现，运行 `evaluate.py` 文件。该文件将从 `Corona_NLP_test.csv` 加载测试数据，并计算包括准确率、精确度、召回率和 F1 分数等评估指标。

```bash
python evaluate.py
```

### 5. **导出模型到 ONNX 格式**

如果您希望将模型导出为 ONNX 格式以便在其他框架中进行推理，可以使用 `export.py` 中的 `export_to_onnx` 函数。此操作将训练好的模型导出到 `covid_model.onnx` 文件。

```python
from export import export_to_onnx

dummy_input_ids = torch.zeros(1, 128, dtype=torch.int64)
dummy_attention_mask = torch.zeros(1, 128, dtype=torch.int64)
dummy_features = torch.zeros(1, 3, dtype=torch.float32)

export_to_onnx(model, (dummy_input_ids, dummy_attention_mask, dummy_features))
```

### 6. **ONNX 推理**

要使用导出的 ONNX 模型进行预测，可以使用 `onnx_predictor.py` 中的 `ONNXTweetPredictor` 类。该类使用 `onnxruntime` 进行推理。

```python
from onnx_predictor import ONNXTweetPredictor

onnx_predictor = ONNXTweetPredictor(onnx_model_path='covid_model.onnx')
print(onnx_predictor.predict("Vaccine distribution is going great!"))
```

---

## 扩展和改进模型

为了进一步提升当前大约 85% 的准确率或构建其他功能，您可以考虑：
- **使用领域特定的预训练模型：**  
  可以尝试使用 [BERTweet](https://github.com/VinAIResearch/BERTweet) 或 [COVID-Twitter-BERT](https://github.com/cdqa-suite/covid-twitter-bert) 等模型。
- **增强手工特征：**  
  添加更多特征（例如情感词典分数、表情符号解析、用户元数据）。
- **多任务学习：**  
  扩展模型来执行其他任务（例如话题检测、讽刺识别）。
- **部署：**  
  使用 Flask 或 FastAPI 创建 Web 应用，实现实时情感分析。
- **数据增强：**  
  使用反向翻译或同义词替换等技术来丰富训练数据集。

---
## 致谢

特别感谢 Hugging Face Transformers 和 PyTorch 社区提供的优秀工具，以及所有为本项目贡献思想和方法的研究人员。

---
