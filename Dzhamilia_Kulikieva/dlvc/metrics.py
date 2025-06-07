from abc import ABCMeta, abstractmethod
import torch
import numpy as np


class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(classes)
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)



    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

        if prediction.dim() != 4 or target.dim() != 3:
            raise ValueError("Shape mismatch: prediction (b,c,h,w), target (b,h,w) required")

        pred_classes = torch.argmax(prediction, dim=1)  # (b,h,w)
        pred_classes = pred_classes.cpu().numpy()
        target_np = target.cpu().numpy()

        for i in range(pred_classes.shape[0]):  # iterate over batch
            pred_flat = pred_classes[i].flatten()
            target_flat = target_np[i].flatten()

            mask = target_flat != 255  # exclude ignored pixels
            pred_filtered = pred_flat[mask]
            target_filtered = target_flat[mask]

            for t, p in zip(target_filtered, pred_filtered):
                if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                    self.conf_matrix[t, p] += 1
   

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"mIoU: {self.mIoU():.2f}"
          

    
    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        ious = []
        for i in range(self.num_classes):
            tp = self.conf_matrix[i, i]
            fp = self.conf_matrix[:, i].sum() - tp
            fn = self.conf_matrix[i, :].sum() - tp
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 0.0
            ious.append(iou)

        if len(ious) == 0:
            return 0.0
        return float(np.mean(ious))





