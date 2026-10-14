# -*- coding: utf-8 -*-
"""
Neural Network 모델 (PyTorch)

특징:
- One-Hot Encoding + Scaling 필요
- GPU 지원
- Early Stopping 지원
"""

import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel


class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # 출력층 (이진 분류)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()


class NeuralNetModel(BaseModel):
    """PyTorch Neural Network 분류 모델"""

    def __init__(self, params: Dict[str, Any], name: str = "NeuralNet"):
        super().__init__(params, name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim: Optional[int] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        cat_features: Optional[list] = None,
    ) -> "NeuralNetModel":
        """
        Neural Network 학습

        Note: X_train은 이미 One-Hot + Scaling이 적용되어 있어야 함
        """
        self.input_dim = X_train.shape[1]

        # 하이퍼파라미터
        hidden_dims = self.params.get("hidden_dims", [256, 128, 64])
        dropout = self.params.get("dropout", 0.3)
        lr = self.params.get("learning_rate", 0.001)
        batch_size = self.params.get("batch_size", 1024)
        epochs = self.params.get("epochs", 100)
        patience = self.params.get("early_stopping_patience", 10)

        # 모델 생성
        self.model = MLP(self.input_dim, hidden_dims, dropout).to(self.device)

        # 데이터 로더
        X_tensor = torch.FloatTensor(X_train.values).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # 학습 설정
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Early Stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # 학습 루프
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                # Early Stopping 체크
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Best model 복원
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """양성 클래스 확률 반환"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        self.model.eval()
        X_tensor = torch.FloatTensor(X.values).to(self.device)

        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy()

        return proba

    def save(self, path: str) -> None:
        """모델 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "params": self.params,
            "name": self.name,
        }, path)

    def load(self, path: str) -> "NeuralNetModel":
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)

        self.input_dim = checkpoint["input_dim"]
        self.params = checkpoint["params"]
        self.name = checkpoint["name"]

        hidden_dims = self.params.get("hidden_dims", [256, 128, 64])
        dropout = self.params.get("dropout", 0.3)

        self.model = MLP(self.input_dim, hidden_dims, dropout).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.is_fitted = True
        return self
