# 選擇語言/choose language
- [中文](README.zh.md)
- [English](README.en.md)

# MNIST-GAN：手寫數字生成

本專案實現了一個生成對抗網絡（GAN）用於生成逼真的手寫數字圖像，基於 MNIST 數據集進行訓練。專案包含生成器和判別器模型，並提供訓練、視覺化和保存結果的程式碼。

---

## 功能特點

- 生成逼真的手寫數字圖像
- 使用 MNIST 數據集進行訓練
- 完全使用 PyTorch 實現
- 可自定義超參數進行實驗
- 保存生成的圖像和訓練後的模型參數
- 包含詳細的中文註釋以便理解

---

## 環境要求

在運行本專案之前，請確保已安裝以下內容：

- Python 3.9 或更高版本

---

## 安裝說明

1. clone 存儲庫：
   ```bash
   git clone https://github.com/sayksii/MNIST-GAN-Handwritten-Digit-Generation.git
   cd MNIST-GAN-Handwritten-Digit-Generation
   ```

2. 安裝所需庫：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. 訓練 GAN 模型：
   ```bash
   python MNIST_GAN.py
   ```

2. 訓練過程中：
   - 將顯示生成器和判別器的損失值
   - 每 500 個批次會在 fake_images 目錄中保存生成的圖像

3. 訓練完成後：
   - 生成器和判別器的權重將分別保存為 `generator.pth` 和 `discriminator.pth`

## 文件結構

```
MNIST-GAN-Handwritten-Digit-Generation/
├── MNIST_GAN.py           # 訓練 GAN 的主要腳本
├── requirements.txt       # 依賴項
├── data/                  # MNIST 數據集目錄（自動下載）
├── fake_images/           # 生成圖像的目錄
└── README.md              # 專案文檔
```

## 模型說明

### 生成器
生成器通過全連接層和活化函數將隨機噪聲映射為逼真的手寫數字圖像。

### 判別器
判別器通過一系列全連接層和 LeakyReLU 活化函數來區分真實圖像和生成圖像。

## 結果展示
在訓練過程中，生成的手寫數字圖像會保存在 `fake_images/` 目錄中。

## 許可證
本專案採用 MIT 許可證。詳情請參見 LICENSE 文件。

## 作者
sayksii([github](https://github.com/sayksii))
