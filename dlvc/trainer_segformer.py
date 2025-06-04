import torch
from pathlib import Path
from typing import Tuple
from abc import ABCMeta, abstractmethod
from tqdm import tqdm

from dlvc.wandb_logger import WandBLogger  # Make sure this is available or disable if needed

print("[DEBUG] Modules imported successfully")


class BaseTrainer(metaclass=ABCMeta):
    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        pass

    @abstractmethod
    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        pass


class SegFormerTrainer(BaseTrainer):
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
                 val_frequency: int = 5,
                 use_wandb: bool = True):

        print("[DEBUG] Initializing SegFormerTrainer...")

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

        print("[DEBUG] Setting up DataLoaders...")
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

        print("[DEBUG] Setting up WandB logger...")
        self.wandb_logger = WandBLogger(enabled=use_wandb, model=self.model, run_name="SegFormer_from_scratch")

        print("[INFO] SegFormerTrainer initialized.")

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        print(f"[DEBUG] Starting training epoch {epoch_idx}")
        self.model.train()
        total_loss = 0.0
        self.train_metric.reset()

        for batch_idx, (images, targets) in enumerate(tqdm(self.train_loader, desc=f"[Train] Epoch {epoch_idx}")):
            images = images.to(self.device)
            targets = targets.squeeze(1).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.train_metric.update(outputs.detach(), targets.detach())

            if batch_idx % 10 == 0:
                print(f"[DEBUG] Batch {batch_idx} — Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        miou = self.train_metric.mIoU()

        print(f"[TRAIN] Epoch {epoch_idx}: Avg Loss = {avg_loss:.4f}, mIoU = {miou:.4f}")
        return avg_loss, miou

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        print(f"[DEBUG] Starting validation epoch {epoch_idx}")
        self.model.eval()
        total_loss = 0.0
        self.val_metric.reset()

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(self.val_loader, desc=f"[Val] Epoch {epoch_idx}")):
                images = images.to(self.device)
                targets = targets.squeeze(1).to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                self.val_metric.update(outputs, targets)

                if batch_idx % 10 == 0:
                    print(f"[DEBUG] Validation batch {batch_idx} — Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(self.val_loader)
        miou = self.val_metric.mIoU()

        print(f"[VAL] Epoch {epoch_idx}: Avg Loss = {avg_loss:.4f}, mIoU = {miou:.4f}")
        return avg_loss, miou

    def train(self) -> None:
        print("[INFO] Starting SegFormer training loop...")
        for epoch in range(self.num_epochs):
            print(f"\n[INFO] Epoch {epoch + 1}/{self.num_epochs}")
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
                    save_path = self.training_save_dir / "best_segformer_model.pt"
                    torch.save(self.model.state_dict(), save_path)
                    print(f"[INFO] Best model saved at epoch {epoch+1} with mIoU: {val_miou:.4f}")

            # ✅ Log to W&B
            print(f"[DEBUG] Logging metrics for epoch {epoch}")
            self.wandb_logger.log(log_dict, step=epoch)

        print("[INFO] Training finished.")
        self.wandb_logger.finish()
