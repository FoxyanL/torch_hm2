import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import logging

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Модель
class LogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

# Генерация данных
def get_data(poly=False, interaction_only=False, stats=False):
    X, y = make_classification(n_samples=500, n_features=5, n_classes=2, random_state=42)    
    features = [X]
    # Полиномиальные признаки
    if poly:
        pf = PolynomialFeatures(degree=2, interaction_only=interaction_only, include_bias=False)
        X_poly = pf.fit_transform(X)
        features.append(X_poly)
    # Статистические признаки
    if stats:
        row_mean = np.mean(X, axis=1, keepdims=True)
        row_std = np.std(X, axis=1, keepdims=True)
        features.append(row_mean)
        features.append(row_std)
    X = np.hstack(features)
    # Нормализация
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)


# Обучение
def train_model(X_train, y_train, X_val, y_val, lr, batch_size, optimizer_name):
    model = LogisticRegression(X_train.shape[1], 2)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown optimizer")

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(20):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        logging.info(f"Epoch {epoch+1}/20 - Loss: {loss.item():.4f}")

    # Оценка
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        preds = torch.argmax(model(X_val_tensor), dim=1).numpy()
    acc = accuracy_score(y_val, preds)
    return acc

# Эксперимент
def run_experiments():
    results = []
    learning_rates = [0.01, 0.05, 0.1]
    batch_sizes = [16, 32, 64]
    optimizers = ['SGD', 'Adam', 'RMSprop']

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for opt in optimizers:
                logging.info(f"Training with lr={lr}, batch_size={batch_size}, optimizer={opt}")
                X_train, X_val, y_train, y_val = get_data()
                acc = train_model(X_train, y_train, X_val, y_val, lr, batch_size, opt)
                results.append((lr, batch_size, opt, acc))
    
    return results

# Визуализация
def plot_results(results):
    import pandas as pd
    df = pd.DataFrame(results, columns=['LR', 'BatchSize', 'Optimizer', 'Accuracy'])
    pivot = df.pivot_table(index='LR', columns='Optimizer', values='Accuracy', aggfunc='max')
    pivot.plot(kind='bar', title='Accuracy by LR and Optimizer')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig("plots/experiment_results.png")
    plt.close()

if __name__ == "__main__":
    results = run_experiments()
    plot_results(results)
