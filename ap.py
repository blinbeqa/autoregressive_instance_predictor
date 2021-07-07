import numpy as np
import random
import torch
import torch.nn.functional as F

class APMetric():
    def __init__(self, threshold = 0.5):
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
    def add_score(self, score, pred_bool, gt_bool):
        if not pred_bool and not gt_bool:
            self.tp = self.tp
            self.tp = self.tp
            self.tn = self.tn
            self.fn = self.fn
        elif not pred_bool and gt_bool:
            self.fn += 1
        else:
            if score > self.threshold:
                self.tp += 1
            else:
                self.fp += 1

    def return_precision(self):
        return self.tp / (self.tp + self.fp)

    def return_recall(self):
        return self.tp / (self.tp + self.fn)


