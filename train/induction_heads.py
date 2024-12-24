import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
from config import MambaConfig_for_induction_heads
from mambapp import MambaLMHeadModel

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Generate Synthetic Data
class InductionDataset(Dataset):
    def __init__(self, num_samples=10000, seq_length=256, vocab_size=16):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.data = self.generate_data()

    def generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # Generate a sequence of half the desired length with random values
            half_seq = [random.randint(1, self.vocab_size-1) for _ in range(self.seq_length // 2)]
            # Repeat the sequence to create an induction pattern
            full_seq = half_seq + half_seq
            # Target sequence is shifted by one position
            target = full_seq[1:] + [0]  # Shifted sequence with padding
            data.append((full_seq, target))
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.tensor(seq), torch.tensor(target)


# 3. Train the Model
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for seq, target in dataloader:
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')


# 4. Evaluation Function
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for seq, target in dataloader:
            seq, target = seq.to(device), target.to(device)
            output = model(seq)
            predictions = output.argmax(dim=-1)
            correct += (predictions == target).sum().item()
            total += target.numel()

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')


# Main
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 16
    num_layers = 2  # 2-layer Transformer model
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 25

    # Create dataset and dataloader
    dataset = InductionDataset(num_samples=65536,seq_length=256, vocab_size=16)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    mambaconfig=MambaConfig_for_induction_heads()
    model = model = MambaLMHeadModel(mambaconfig, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and evaluate the model
    train_model(model, dataloader, criterion, optimizer, num_epochs)
    evaluate_model(model, dataloader)
