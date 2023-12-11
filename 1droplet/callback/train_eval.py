import torch
import torch.nn as nn
import torch.optim as optim

class TrainEval:
    """ Train and evaluate a model
    
    Note:
        Do not include `self` param in ``Args`` section.
    
    Args:
        model (torch.nn.Module): Model to train and evaluate.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        clip (float): Gradient clipping value. Default: -1 (if the value is negative, gradient clipping is not performed.)
        
    Examples:
        >>> train_eval = TrainEval(model, train_loader, val_loader, criterion, optimizer, clip = -1)
        >>> loss_list = train_eval.train(num_epochs)

    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, clip = -1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.clip = clip

    def train(self, num_epochs):
        self.loss = {
            'train': [],
            'val': []
        }
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0

            for x, t in self.train_loader:
                self.optimizer.zero_grad() 
                outputs = self.model(x) # prediction
                loss = self.criterion(outputs, t) # compare prediction and correct data
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader) # バッチサイズによるロスの平均
            
            # validation
            val_loss, val_accuracy = self.evaluate()
            
            # lossを格納
            self.loss['train'].append(train_loss)
            self.loss['val'].append(val_loss)
            
            if epoch == 0 or (epoch+1) % 20 == 0:
                print("[{}/{}] \t Train Loss: {:.4f} \t Validation Loss: {:.4f} \t Validation Accuracy: {:.4f}"
                    .format(epoch+1, num_epochs, train_loss, val_loss, val_accuracy))
        
        return self.loss
    
    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss /= len(self.val_loader)
        val_accuracy = correct / total
        
        return val_loss, val_accuracy