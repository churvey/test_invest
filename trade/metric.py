
import torchmetrics
from torchmetrics import AUROC, Accuracy, PrecisionRecallCurve,F1Score,Precision,Recall

from torchmetrics.regression import PearsonCorrCoef,SpearmanCorrCoef,R2Score,MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError
import numpy as np
import torch
                
                
class Metric:
    def __init__(self, writer, stage, roll, metrics = {
        "pcorr":PearsonCorrCoef().to("cuda"),
            # "scorr": SpearmanCorrCoef().to("cuda"),
            # "mse": MeanSquaredError().to("cuda"),
            # "mae": MeanAbsoluteError().to("cuda"),
            # "mape":MeanAbsolutePercentageError().to("cuda"),
            # "r2" :R2Score().to("cuda")
            # "auc": AUROC(task="multiclass", num_classes=3).to("cuda"),
            # "acc": Accuracy(task="multiclass", num_classes=3).to("cuda"),
            # # "prc":PrecisionRecallCurve(task="multiclass", num_classes=3).to("cuda"),
            # "f1":F1Score(task="multiclass", num_classes=3).to("cuda"),
            # "recall":F1Score(task="multiclass", num_classes=3).to("cuda"),
            # "p_1":Precision(task="multiclass", num_classes=3,   ).to("cuda"),
            # "p_2":Precision(task="multiclass", num_classes=3,  average='micro' ).to("cuda"),
            # "p_3":Precision(task="multiclass", num_classes=3,  average='weighted' ).to("cuda"),
            # "r_1":Recall(task="multiclass", num_classes=3,   ).to("cuda"),
            # "r_2":Recall(task="multiclass", num_classes=3,  average='micro' ).to("cuda"),
            # "r_3":Recall(task="multiclass", num_classes=3,  average='weighted' ).to("cuda"),
    }):
        self.writer = writer
        self.stage = stage
        self.roll = roll
        self.ys = []
        self.yhats = []
        self.summary_step = 1
        self.metrics = metrics if metrics else {}
        
    def metric(self, name):
        return f"{name}/{self.stage}/roll-{self.roll}"
    
    def update(self, y, yhat, step):

        if step % self.summary_step == 0:
            if "pcorr" in self.metrics:
                self.writer.add_histogram(
                    self.metric("y_dis"),
                    y.cpu().numpy(),
                    step
                )
                self.writer.add_histogram(
                    self.metric("yhat_dis"),
                    yhat.cpu().numpy(),
                    step
                )
                y_p = torch.sum(yhat > 0).cpu().numpy() * 1.0 / yhat.shape[0]
                m = {}
                m.update(
                    {"y_pos": y_p}
                )
                for k, v in m.items():
                    self.writer.add_scalar(self.metric(f"{k}-step"), v, step)
            # print(yhat.shape, y.shape)
            for k, v in self.metrics.items():
                self.writer.add_scalar(self.metric(f"{k}-step"), v(yhat, y).cpu().numpy(), step)
        
        
    def finish(self, epoch, **kvargs):
    
        for k, v in self.metrics.items():
            self.writer.add_scalar(self.metric(f"{k}"), v.compute().cpu().numpy(), epoch)
            v.reset()
            
        for k, v in kvargs.items():
            if isinstance(v, dict):
                self.writer.add_scalars(self.metric(f"{k}"), v, epoch)
            else:
                self.writer.add_scalar(self.metric(f"{k}"), v, epoch)
        
        # if self.metric:
        #     print(f"finish {self.stage} roll-{self.roll} epoch-{epoch}")
        