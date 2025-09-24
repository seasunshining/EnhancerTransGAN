import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import optuna
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
skip_organizations = [
    "Human_Activated_IL4_anti-CD40_IgM_and_IgD_peripheral_blood",
    "Human_Blood_vessel",
    "Human_Cytotrophoblast",
    "Human_Endometrial_epithelial",
    "Human_Pancreatic",
    "Human_Peyers_patch",
    "Human_Tongue"
]

data_folder_path = '/work/38liCQ/each-tissue/'
model_save_base_path = '/each-tissue-prediction/prediction-model-save'
tissue_folders = [f for f in os.listdir(data_folder_path) if f.endswith('_hg38_TE')]

tissue_folders = [f for f in tissue_folders if f not in skip_organizations]

split_index = len(tissue_folders) // 2
tissue_folders_4090 = tissue_folders[:split_index]
tissue_folders_ada6000 = tissue_folders[split_index:]

def filter_sequences(sequences, labels):
    valid_chars = set('ATCG')
    filtered_sequences = []
    filtered_labels = []
    
    for seq, label in zip(sequences, labels):
        if all(char in valid_chars for char in seq):
            filtered_sequences.append(seq)
            filtered_labels.append(label)
    
    return np.array(filtered_sequences), np.array(filtered_labels)  # 确保返回的是numpy数组

def encode_sequences(sequences, tokenizer, max_length):
    encoded_sequences = [tokenizer.encode(seq, add_special_tokens=True, max_length=max_length, truncation=True, padding='max_length') for seq in sequences]
    return np.array(encoded_sequences)

class BertRegressionModel(nn.Module):
    def __init__(self, hidden_dim):
        super(BertRegressionModel, self).__init__()
        self.bert = AutoModel.from_pretrained(
            '/each-tissue-prediction/dnabert2', 
            trust_remote_code=True
        )
        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        cls_token_output = last_hidden_state[:, 0, :]
        x = torch.relu(self.fc1(cls_token_output))
        x = self.fc2(x)
        return x


def objective(trial, train_dataloader, val_dataloader):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=32)
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

    model = BertRegressionModel(hidden_dim)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    scaler = GradScaler()
    accumulation_steps = 4

    best_val_loss = float('inf')
    patience, patience_threshold = 0, 5

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)

            with autocast():
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, labels = batch
                input_ids, labels = input_ids.to(device), labels.to(device)
                with autocast():  # 混合精度推理
                    outputs = model(input_ids)
                    val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch+1} - Training Loss: {running_loss / len(train_dataloader)} - Validation Loss: {val_loss}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience > patience_threshold:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_val_loss

def tune_model(sequences, labels, max_length, tuner_epochs=25):
    tokenizer = AutoTokenizer.from_pretrained('/each-tissue-prediction/dnabert2', trust_remote_code=True)

    encoded_sequences = encode_sequences(sequences, tokenizer, max_length)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(torch.tensor(encoded_sequences, dtype=torch.long), labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_dataloader, val_dataloader), n_trials=tuner_epochs)

    best_trial = study.best_trial
    print(f"Best hyperparameters: {best_trial.params}")

    return best_trial

def inverse_transform(predictions, scaler):
    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

for tissue_folder in tissue_folders_4090:
    tissue_name = tissue_folder.replace('_hg38_TE', '')
    data_path = os.path.join(data_folder_path, tissue_folder, f'{tissue_name}_train.csv')
    
    data = pd.read_csv(data_path)
    sequences = data['Sequence'].tolist()
    labels = data['te_cas_value'].values
    
    sequences, labels = filter_sequences(sequences, labels)

    scaler = MinMaxScaler()
    labels = labels.reshape(-1, 1)
    labels_normalized = scaler.fit_transform(labels).flatten()

    max_length = 500

    best_trial = tune_model(sequences, labels_normalized, max_length)

    hidden_dim = best_trial.params['hidden_dim']
    learning_rate = best_trial.params['lr']
    model = BertRegressionModel(hidden_dim)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    tokenizer = AutoTokenizer.from_pretrained('/each-tissue-prediction/dnabert2', trust_remote_code=True)

    encoded_sequences = encode_sequences(sequences, tokenizer, max_length)
    labels = torch.tensor(labels_normalized, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(torch.tensor(encoded_sequences, dtype=torch.long), labels)
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    scaler = GradScaler()
    for epoch in range(50):
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            with autocast():
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model_save_path = os.path.join(model_save_base_path, f'{tissue_name}-best_model-bert-te_cas_prediction.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model for {tissue_name} saved to {model_save_path}')
