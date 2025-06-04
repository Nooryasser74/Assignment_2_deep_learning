# import collections
# import torch
# from typing import  Tuple
# from abc import ABCMeta, abstractmethod
# from pathlib import Path
# from tqdm import tqdm

# #from dlvc.wandb_logger import WandBLogger

# class BaseTrainer(metaclass=ABCMeta):
#     '''
#     Base class of all Trainers.
#     '''

#     @abstractmethod
#     def train(self) -> None:
#         '''
#         Returns the number of samples in the dataset.
#         '''

#         pass

#     @abstractmethod
#     def _val_epoch(self) -> Tuple[float, float]:
#         '''
#         Returns the number of samples in the dataset.
#         '''

#         pass

#     @abstractmethod
#     def _train_epoch(self) -> Tuple[float, float]:
#         '''
#         Returns the number of samples in the dataset.
#         '''

#         pass

# class ImgSemSegTrainer(BaseTrainer):
#     """
#     Class that stores the logic for training a model for image classification.
#     """
#     def __init__(self, 
#                  model, 
#                  optimizer,
#                  loss_fn,
#                  lr_scheduler,
#                  train_metric,
#                  val_metric,
#                  train_data,
#                  val_data,
#                  device,
#                  num_epochs: int, 
#                  training_save_dir: Path,
#                  batch_size: int = 4,
#                  val_frequency: int = 5):
#         '''
#         Args and Kwargs:
#             model (nn.Module): Deep Network to train
#             optimizer (torch.optim): optimizer used to train the network
#             loss_fn (torch.nn): loss function used to train the network
#             lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
#             train_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of training set
#             val_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of validation set
#             train_data (dlvc.datasets...): Train dataset
#             val_data (dlvc.datasets...): Validation dataset
#             device (torch.device): cuda or cpu - device used to train the network
#             num_epochs (int): number of epochs to train the network
#             training_save_dir (Path): the path to the folder where the best model is stored
#             batch_size (int): number of samples in one batch 
#             val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th 
#                                 epoch we evaluate model on validation set)

#         What does it do:
#             - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
#             - Creates data loaders for the train and validation datasets
#             - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

#         '''
        

    
#         ##TODO implement
#         # recycle your code from assignment 1 or use/adapt reference implementation
#         pass
        

#     def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
#         """
#         Training logic for one epoch. 
#         Prints current metrics at end of epoch.
#         Returns loss, mean IoU for this epoch.

#         epoch_idx (int): Current epoch number
#         """
#         ##TODO implement
#         # recycle your code from assignment 1 or use/adapt reference implementation
#         pass


#     def _val_epoch(self, epoch_idx:int) -> Tuple[float, float]:
#         """
#         Validation logic for one epoch. 
#         Prints current metrics at end of epoch.
#         Returns loss, mean IoU for this epoch on the validation data set.

#         epoch_idx (int): Current epoch number
#         """
#         ##TODO implement
#         # recycle your code from assignment 1 or use/adapt reference implementation
#         pass

#     def train(self) -> None:
#         """
#         Full training logic that loops over num_epochs and
#         uses the _train_epoch and _val_epoch methods.
#         Save the model if mean IoU on validation data set is higher
#         than currently saved best mean IoU or if it is end of training. 
#         Depending on the val_frequency parameter, validation is not performed every epoch.
#         """
#         ##TODO implement
#         # recycle your code from assignment 1 or use/adapt reference implementation
#         pass

                





            
            
import collections
import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

from dlvc.wandb_logger import WandBLogger  

class BaseTrainer(metaclass=ABCMeta):
    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float]:
        pass


class ImgSemSegTrainer(BaseTrainer):
    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_metric,
                 val_metric,
                 train_data,
                 val_data,
                 device,
                 num_epochs: int,
                 training_save_dir: Path,
                 batch_size: int = 4,
                 val_frequency: int = 5):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.best_miou = 0.0

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

        print("[DEBUG] Trainer initialized. DataLoaders created.")

        # ✅ Initialize W&B logger
        self.wandb_logger = WandBLogger(enabled=True, model=self.model, run_name="fcn from scratch")

        # i had also ano 
    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        self.train_metric.reset()
        print(f"[DEBUG] Starting training epoch {epoch_idx}")

        for i, (images, targets) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            targets = targets.squeeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(images)["out"]
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.train_metric.update(outputs.detach(), targets.detach())

        avg_loss = total_loss / len(self.train_loader)
        miou = self.train_metric.mIoU()
        print(f"[TRAIN] Epoch {epoch_idx}: Avg Loss = {avg_loss:.4f}, mIoU = {miou:.4f}")
        return avg_loss, miou

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        self.val_metric.reset()
        print(f"[DEBUG] Starting validation epoch {epoch_idx}")

        with torch.no_grad():
            for i, (images, targets) in enumerate(self.val_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                targets = targets.squeeze(1)

                outputs = self.model(images)["out"]
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                self.val_metric.update(outputs, targets)

        avg_loss = total_loss / len(self.val_loader)
        miou = self.val_metric.mIoU()
        print(f"[VAL] Epoch {epoch_idx}: Avg Loss = {avg_loss:.4f}, mIoU = {miou:.4f}")
        return avg_loss, miou

    def train(self) -> None:
        print("[DEBUG] Starting full training loop...")
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            train_loss, train_miou = self._train_epoch(epoch)
            self.lr_scheduler.step()

            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/mIoU": train_miou
            }

            if (epoch + 1) % self.val_frequency == 0 or (epoch + 1) == self.num_epochs:
                val_loss, val_miou = self._val_epoch(epoch)
                log_dict.update({
                    "val/loss": val_loss,
                    "val/mIoU": val_miou
                })

                if val_miou > self.best_miou:
                    self.best_miou = val_miou
                    save_path = self.training_save_dir / "best_model.pt"
                    torch.save(self.model.state_dict(), save_path)
                    print(f"[INFO] Best model saved with mIoU: {val_miou:.4f}")

            # ✅ Log metrics to W&B
            self.wandb_logger.log(log_dict, step=epoch)


        print("[INFO] Training finished.")
        self.wandb_logger.finish()
