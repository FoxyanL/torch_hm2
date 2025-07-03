import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn as nn
import torch.optim as optim
from homework_model_modification import LinearRegression, LogisticRegressionMulticlass
import logging
from utils import log_epoch


class CSVDataset(Dataset):
    def __init__(self, csv_file, target_column, task='regression'):
        self.df = pd.read_csv(csv_file)

        self.y = self.df[target_column]
        self.X = self.df.drop(columns=[target_column])

        # Кодирование категориальных признаков
        for col in self.X.select_dtypes(include='object').columns:
            self.X[col] = LabelEncoder().fit_transform(self.X[col])

        # Нормализация числовых признаков
        self.X = pd.DataFrame(StandardScaler().fit_transform(self.X))

        # Обработка целевой переменной
        if task == 'classification':
            if self.y.dtype == 'object' or self.y.nunique() <= 2:
                self.y = LabelEncoder().fit_transform(self.y)
            self.y = torch.tensor(self.y, dtype=torch.long)
        else:
            self.y = torch.tensor(self.y.values, dtype=torch.float32)

        self.X = torch.tensor(self.X.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_regression(csv_path, target_column):
    """
    Регрессия на boston.csv
    """
    dataset = CSVDataset(csv_path, target_column, task='regression')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LinearRegression(in_features=dataset.X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 51):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        log_epoch(epoch, avg_loss)
        logging.info(f"Epoch {epoch}/50 - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'models/regression_model.pth')
    logging.info("Модель регрессии натренирована и сохранена")

def train_classification(csv_path, target_column):
    dataset = CSVDataset(csv_path, target_column, task='classification')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_classes = len(torch.unique(dataset.y))

    model = LogisticRegressionMulticlass(in_features=dataset.X.shape[1], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 51):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        log_epoch(epoch, avg_loss)
        logging.info(f"Epoch {epoch}/50 - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'models/classification_model.pth')
    logging.info("Модель классификации натренирована и сохранена")

if __name__ == '__main__':
    train_regression('data/boston.csv', target_column='medv')
    train_classification('data/titanic.csv', target_column='Survived')
