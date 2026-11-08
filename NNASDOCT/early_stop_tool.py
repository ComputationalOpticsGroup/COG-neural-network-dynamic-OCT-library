"""Early stopping adapted from Bjarten/early-stopping-pytorch (MIT).

Original source: https://github.com/Bjarten/early-stopping-pytorch
Copyright (c) 2018 Bjarte Mehus Sunde; see LICENSE.
"""

import os
import numpy as np


class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, root, verbose=True, delta=0, trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.trace_func = trace_func
        self.path = os.path.join(root, "model.h5")

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # Equal validation loss also counts as non-improvement.
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss improved ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save(self.path)
        self.val_loss_min = val_loss
