# src/lstm_model.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def create_sequences(data, seq_len):
    """
    Create sliding window sequences.
    X: (n_samples, seq_len, 1)
    y: (n_samples, 1)
    """
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs)[..., np.newaxis], np.array(ys)[..., np.newaxis]

class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(LSTMAnomalyDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, train_loader, num_epochs=20, learning_rate=0.001, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
            
    return model

def predict_lstm(model, data, seq_len, device='cpu'):
    """
    Predict on new data. Data should be the full series (including context).
    Returns predictions aligned with data[seq_len:].
    """
    model.eval()
    sequences, _ = create_sequences(data, seq_len)
    dataset = TimeSeriesDataset(sequences, np.zeros(len(sequences))) # Dummy targets
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    preds = []
    with torch.no_grad():
        for seqs, _ in loader:
            seqs = seqs.to(device)
            outputs = model(seqs)
            preds.append(outputs.cpu().numpy())
            
    return np.concatenate(preds).flatten()
