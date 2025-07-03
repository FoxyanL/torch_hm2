import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

from utils import (
    make_regression_data, make_classification_data, log_epoch,
    RegressionDataset, ClassificationDataset
)

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Линейная регрессия с L1/L2 и early stopping
class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

def train_linear_regression():
    logging.info("Генерация данных для линейной регрессии...")
    X, y = make_regression_data(n=200)
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LinearRegression(in_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    l1_lambda = 0.01
    l2_lambda = 0.01
    patience = 10
    min_delta = 1e-4
    best_loss = float('inf')
    counter = 0

    for epoch in range(1, 101):
        total_loss = 0
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            l1 = sum(torch.sum(torch.abs(param)) for param in model.parameters())
            l2 = sum(torch.sum(param ** 2) for param in model.parameters())
            loss += l1_lambda * l1 + l2_lambda * l2

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (i + 1)
        log_epoch(epoch, avg_loss)
        logging.info(f"Epoch {epoch}/100 - Loss: {avg_loss:.4f}")

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break

    torch.save(model.state_dict(), 'models/linreg_mod.pth')
    logging.info("Модель линейной регрессии сохранена в 'models/linreg_mod.pth'")


# Логистическая регрессия и confusion matrix
class LogisticRegressionMulticlass(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

def train_logistic_regression():
    logging.info("Генерация данных для мультиклассовой логистической регрессии...")
    X, y = make_classification_data(n=300, source='synthetic_multiclass')
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LogisticRegressionMulticlass(in_features=X.shape[1], num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    all_preds = []
    all_true = []

    for epoch in range(1, 101):
        total_loss = 0
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / (i + 1)
        log_epoch(epoch, avg_loss)
        logging.info(f"Epoch {epoch}/100 - Loss: {avg_loss:.4f}")

        if epoch % 10 == 0:
            p = precision_score(all_true, all_preds, average='macro')
            r = recall_score(all_true, all_preds, average='macro')
            f1 = f1_score(all_true, all_preds, average='macro')
            try:
                roc_auc = roc_auc_score(
                    np.eye(3)[all_true],
                    np.eye(3)[all_preds],
                    multi_class='ovr'
                )
            except ValueError:
                roc_auc = float('nan')

            logging.info(f"Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")

        all_preds.clear()
        all_true.clear()

    torch.save(model.state_dict(), 'models/logreg_multiclass.pth')
    logging.info("Модель логистической регрессии сохранена в 'models/logreg_multiclass.pth'")

    # Confusion matrix
    model.eval()
    X_tensor = X.detach().clone().float()
    logits = model(X_tensor)
    preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).numpy()

    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png")
    plt.close()
    logging.info("Confusion matrix сохранена в 'plots/confusion_matrix.png'")


if __name__ == '__main__':
    logging.info("Обучение линейной регрессии с регуляризацией и early stopping")
    train_linear_regression()
    logging.info("Обучение логистической регрессии с мультиклассом и метриками")
    train_logistic_regression()
