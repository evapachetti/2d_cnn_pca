# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.trace_func = trace_func


    def __call__(self, val_loss, train_loss):

        score_val = val_loss
        score_train = train_loss

        if score_val > score_train:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

    