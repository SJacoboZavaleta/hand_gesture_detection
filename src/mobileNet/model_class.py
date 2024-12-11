import thop
import torch
import torch.nn.functional as F
import torchmetrics #conda install -c conda-forge torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import time
import json
import torchviz
import graphviz
from torchsummary import summary

class ASLModel(nn.Module):
    """
    Custom Neural Network model for ASL gesture recognition using transfer learning.

    This model uses MobileNetV2 as the backbone with custom classification layers.
    The architecture is designed to balance accuracy and computational efficiency.

    Architecture Overview:
    ---------------------
    1. MobileNetV2 backbone (pretrained on ImageNet)
    2. Custom dense layers with batch normalization
    3. Dropout for regularization
    4. Softmax output layer

    Attributes:
        base_model (nn.Module): Pretrained MobileNetV2 model
        classifier (nn.Sequential): Custom classification layers
        num_classes (int): Number of output classes
    """



    def __init__(self, num_classes=29, base_model_name='mobilenet_v2', dense_units=256, dropout_rate=0.5):
        """
        Initialize the ASL Model.
        Args:
            num_classes (int): Number of output classes (ASL gestures)
            base_model_name (str): Name of the pretrained model to use
            dense_units (int):
            droput_rate (int):

        """
        super().__init__()

        # Load pretrained model 
        if base_model_name.lower() == 'mobilenet_v2':
            # Load MobileNet V2
            self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            #self.base_model = models.mobilenet_v2(pretrained=True)
            # Remove the last classification layer
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
            # Determine the number of features (adjust as needed)
            num_features = 1280
        else:
            raise ValueError(f"Model {base_model_name} not supported")

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Own classifier

        self.dense_block1 = nn.Sequential(
            nn.Linear(num_features, dense_units*2, bias=False),
            nn.BatchNorm1d(dense_units*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.dense_block2 = nn.Sequential(
            nn.Linear(dense_units*2, dense_units, bias=False),
            nn.BatchNorm1d(dense_units),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Modify classifier
        self.classifier = nn.Linear(dense_units, num_classes)

        # Initialize weights
        self._initialize_weights()

        # Freeze pretrained newtork
        self.freeze_base_model()

    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    

    def freeze_base_model(self):
        # Freeze early layers
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, num_layers=30):
        trainable_layers = list(self.base_model.parameters())[-num_layers:]
        for param in trainable_layers:
            param.requires_grad = True



    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, num_classes)
        """

        # Base model feature
        x = self.base_model(x)

        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Dense blocks
        x = self.dense_block1(x)
        x = self.dense_block2(x)

        # Output with softmax
        x = self.classifier(x)

        # We don't apply a classification here, we do it later
        #out = F.softmax(x, dim=1)
        return x