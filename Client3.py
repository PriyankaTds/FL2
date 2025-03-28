import flwr as fl
from utils2 import load_imdb_data, preprocess_data, create_model, split_data
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm  # Import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load IMDb dataset
dataset = load_imdb_data()

# Create model, tokenizer, optimizer, and loss function
model, tokenizer, optimizer, loss_fn = create_model()

# Move model to GPU
model.to(device)

# Preprocess data
train_dataset, test_dataset = preprocess_data(dataset, tokenizer)

# Split data for Client 1
train_subset = split_data(train_dataset, client_id=2)
test_subset = split_data(test_dataset, client_id=2)

# Create DataLoader for Client 1
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_subset, batch_size=16, shuffle=False, num_workers=4)

# Create Flower client
class IMDBClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, optimizer, loss_fn):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scaler = GradScaler()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Set model parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        # Train the model with tqdm progress bar
        self.model.train()
        total_loss, total_accuracy = 0, 0
        num_samples = 0

        # Initialize progress bar
        progress_bar = tqdm(self.train_loader, desc="Training", unit="batch")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Mixed precision training
            with autocast():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs.logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Calculate metrics
            predictions = torch.argmax(outputs.logits, dim=1)
            batch_accuracy = (predictions == labels).sum().item()
            total_accuracy += batch_accuracy
            total_loss += loss.item()
            num_samples += len(labels)

            # Update progress bar description
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{batch_accuracy/len(labels):.2%}"
            })

        # Calculate averages
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_accuracy / num_samples

        print(f"\nTraining Summary - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        return self.get_parameters(config), num_samples, {
            "accuracy": avg_accuracy,
            "num_examples": num_samples,
            "loss": avg_loss
        }

    def evaluate(self, parameters, config):
        # Set model parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        # Evaluate with tqdm progress bar
        self.model.eval()
        total_loss, total_accuracy = 0, 0
        num_samples = 0

        # Initialize progress bar
        progress_bar = tqdm(self.test_loader, desc="Evaluating", unit="batch")
        
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs.logits, labels)
                
                predictions = torch.argmax(outputs.logits, dim=1)
                batch_accuracy = (predictions == labels).sum().item()
                
                total_loss += loss.item()
                total_accuracy += batch_accuracy
                num_samples += len(labels)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{batch_accuracy/len(labels):.2%}"
                })

        # Calculate averages
        avg_loss = total_loss / len(self.test_loader)
        avg_accuracy = total_accuracy / num_samples

        print(f"\nEvaluation Summary - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        return avg_loss, num_samples, {
            "accuracy": avg_accuracy,
            "num_examples": num_samples
        }

# Start Flower client
if __name__ == "__main__":
    client = IMDBClient(model, train_loader, test_loader, optimizer, loss_fn)
    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080",
        client=client
    )