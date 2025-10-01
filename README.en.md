# 選擇語言/choose language
- [中文](README.zh.md)
- [English](README.en.md)

# MNIST-GAN: Handwritten Digit Generation

This project implements a Generative Adversarial Network (GAN) to generate realistic handwritten digit images based on the MNIST dataset. The project includes both the generator and discriminator models and provides code for training, visualization, and saving results.

---

## Features

- Generates realistic handwritten digit images.
- Uses the MNIST dataset for training.
- Fully implemented in PyTorch.
- Customizable hyperparameters for experimentation.
- Saves generated images and trained model parameters.
- Includes comprehensive Chinese comments for better understanding.

---

## Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.9 or above

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sayksii/MNIST-GAN-Handwritten-Digit-Generation.git
   cd MNIST-GAN-Handwritten-Digit-Generation
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
   
## Usage

1. Train the GAN model:
    ```bash
    python MNIST_GAN.py
    ```
2. During training:
   - The generator and discriminator losses will be displayed.
   - Generated images will be saved in the fake_images directory every 500 batches.

3. After training:
   - The generator and discriminator weights will be saved as `generator.pth` and `discriminator.pth`.

## File Structure

```
MNIST-GAN-Handwritten-Digit-Generation/
├── MNIST_GAN.py           # Main script for training the GAN
├── requirements.txt       # Dependencies
├── data/                  # Directory for MNIST dataset (automatically downloaded)
├── fake_images/           # Directory for generated images
└── README.md              # Project documentation
```

## Models

### Generator
The generator maps random noise to realistic handwritten digit images using fully connected layers and activation functions.

### Discriminator
The discriminator distinguishes real images from fake ones using a series of fully connected layers and LeakyReLU activations.

## Results
Generated handwritten digit images are saved during training in the `fake_images/` directory. Example output:

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Author
sayksii([github](https://github.com/sayksii))
