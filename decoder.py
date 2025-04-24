import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Define a simple MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def train_evaluate_mlp(X_train, y_train, X_test, y_test, input_dim, output_dim, hidden_dim=128, num_epochs=10, batch_size=256, lr=0.001):
    """Trains and evaluates the MLP model."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"MLP using device: {device}")

    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = MLP(input_dim, hidden_dim, output_dim).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'MLP Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

    # Evaluate the model
    model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
    
    y_pred = np.array(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f'MLP Accuracy: {accuracy:.4f}')
    print("MLP Classification Report:")
    print(report)
    return accuracy, report

def train_evaluate_logistic_regression(X_train, y_train, X_test, y_test, **kwargs):
    """Trains and evaluates a Logistic Regression model."""
    print("Training Logistic Regression...")
    # Scale data for better performance
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, **kwargs))
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f'Logistic Regression Accuracy: {accuracy:.4f}')
    print("Logistic Regression Classification Report:")
    print(report)
    return accuracy, report

def train_evaluate_svm(X_train, y_train, X_test, y_test, kernel='rbf', C=1.0, gamma='scale', **kwargs):
    """Trains and evaluates an SVM model with a non-linear kernel."""
    print("Training SVM...")
    # Scale data for better performance
    pipeline = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma, **kwargs))
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f'SVM ({kernel} kernel) Accuracy: {accuracy:.4f}')
    print(f"SVM ({kernel} kernel) Classification Report:")
    print(report)
    return accuracy, report 