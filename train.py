import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from model import Transformer
import lasso_dataset

def train_model(model, metrics, epochs, train_loader, valid_loader):
  lr = 1e-3
  num_epochs = epochs
  criterion = nn.CrossEntropyLoss()
  trained_model = model
  optimizer= torch.optim.Adam(model.parameters(), lr)
  loss_metrics = metrics
  for epoch in range(num_epochs):


    #Training
    trained_model.train()
    training_loss = 0
    for x, y in tqdm(train_loader, desc = "Training"):
      optimizer.zero_grad()
      logits = model(x, tgt_mask)[:,-1,:]
      loss = criterion(logits, y)
      training_loss += loss.item()
      loss.backward()
      optimizer.step()
    loss_metrics['training_losses'].append(training_loss/ len(train_loader))

    #Validataion
    trained_model.eval()
    validation_loss = 0
    with torch.no_grad():
      correct = 0
      total = 0
      for x, y in tqdm(valid_loader, desc = "Validation"):
        logits = trained_model(x, tgt_mask)[:, -1, :]
        loss = criterion(logits, y)
        validation_loss += loss.item()
        pred = logits.argmax(dim = -1)
        correct += (pred == y).sum().item()
        total += len(y)
      loss_metrics['validation_accuracy'].append(correct/total)
    loss_metrics['validation_losses'].append(training_loss/ len(valid_loader))
    print(f'Epoch {epoch} - training_loss: {training_loss/len(train_loader):.4f}, validation_loss: {validation_loss/len(valid_loader):.4f}, validation_accuracy: {correct/total :.3f}')


softmax_train_loader, softmax_valid_loader, lasso_train_loader, lasso_valid_loader = lasso_dataset.get_data()
# x, y = next(iter(softmax_train_loader))

# print(x[0])
# print()
# print(f'The corresponding value of inquired key token, {x[0][-1]} : {y[0]}')



vocab_size = 2005
d_model = 128
num_heads = 2
num_layers = 2
max_seq_len = 20
batch = 32
tgt_mask = torch.triu(torch.ones(18, 18), diagonal=1).bool()


softmax_loss_metrics = {
    "training_losses": [],
    'validation_losses': [],
    'validation_accuracy': []
}


lasso_loss_metrics = {
    "training_losses": [],
    'validation_losses': [],
    'validation_accuracy': []
}


lasso_model = Transformer(vocab_size, d_model, num_heads, num_layers, max_seq_len, attention = 'lasso')
softmax_model = Transformer(vocab_size, d_model, num_heads, num_layers, max_seq_len, attention = 'softmax')


train_model(softmax_model, softmax_loss_metrics, 10, softmax_train_loader, softmax_valid_loader)
train_model(lasso_model, lasso_loss_metrics, 10, lasso_train_loader, lasso_valid_loader)




