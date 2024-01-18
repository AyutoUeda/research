class EarlyStopping:
    """earlystoppingクラス
    
    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        verbose (int): If ``True``, prints a message for each validation loss improvement. Default: ``False``.
    """

    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float("inf")
        self.patience = patience
        self.verbose = verbose
        
    def __call__(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print("early stopping")
                return True

        else:
            self._step = 0
            self._loss = loss

        return False