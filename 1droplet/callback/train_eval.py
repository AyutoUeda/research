import torch
import torch.nn as nn
import torch.optim as optim

class TrainEval:
    """ Train and evaluate a model
    trainとevaluateを同時に実行するメソッドを持つクラス
    
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
        >>> loss_dict = train_eval.train(num_epochs)

    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, clip = -1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.clip = clip

    def train(self, num_epochs) -> dict:
        self.loss = {
            'train': [],
            'val': []
        }
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0

            for x, target in self.train_loader:
                self.optimizer.zero_grad() 
                outputs = self.model(x) # prediction
                loss = self.criterion(outputs, target) # compare prediction and correct data
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader) # バッチサイズごとのロスの平均
            
            # validation
            val_loss = self.evaluate()
            
            # lossを格納
            self.loss['train'].append(train_loss)
            self.loss['val'].append(val_loss)
            
            if epoch == 0 or (epoch+1) % 20 == 0:
                print("[{}/{}] \t Train Loss: {:.5f} \t Validation Loss: {:.5f}"
                    .format(epoch+1, num_epochs, train_loss, val_loss))
        
        return self.loss
    
    def evaluate(self) -> float:
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(self.val_loader)
        
        return val_loss