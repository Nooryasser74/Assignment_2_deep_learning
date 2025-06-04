from abc import ABCMeta, abstractmethod
import torch

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

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.conf_matrix = torch.zeros((self.classes, self.classes), dtype=torch.int64)


    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

        if prediction.ndim != 4 or target.ndim != 3:
            raise ValueError("Expected prediction shape (b, c, h, w) and target shape (b, h, w)")

        pred_labels = torch.argmax(prediction, dim=1)  # shape (b, h, w)

        for p, t in zip(pred_labels, target):
            mask = t != 255
            p = p[mask]
            t = t[mask]
            if p.numel() == 0:
                continue
            hist = torch.bincount(
                self.classes * t + p,
                minlength=self.classes ** 2
            ).reshape(self.classes, self.classes)
            self.conf_matrix += hist

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
        if self.conf_matrix.sum() == 0:
            return 0.0

        intersection = torch.diag(self.conf_matrix)
        union = self.conf_matrix.sum(1) + self.conf_matrix.sum(0) - intersection
        iou = torch.where(union == 0, torch.zeros_like(intersection, dtype=torch.float), intersection.float() / union.float())
        return iou.mean().item()





