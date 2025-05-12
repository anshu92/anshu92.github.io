---
title: "CNN for Image Classification (CIFAR-10)"
date: 2025-03-20
draft: false
tags: ["Computer Vision", "CNN", "PyTorch", "Portfolio"]
categories: ["Projects"]
summary: "Built and trained a Convolutional Neural Network using PyTorch to classify images from the CIFAR-10 dataset."
thumbnail: "/images/cnn-project.png" # Add to static/images/
# Optional: Add links using Blowfish's link shortcode or markdown
github_link: "https://github.com/your-username/cnn-cifar10"
# live_demo_link: "..."
---

## Project Goal
The objective was to implement a CNN from scratch...

## Dataset
Used the popular CIFAR-10 dataset containing 60,000 32x32 color images in 10 classes.

## Model Architecture
Designed a standard CNN with convolutional layers, ReLU activations, max-pooling, and fully connected layers.

```python
# Simplified PyTorch model definition
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # ... more layers
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # ... forward pass logic
        return x