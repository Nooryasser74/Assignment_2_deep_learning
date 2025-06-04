# import torch
# import torch.nn as nn
# import torch.optim as optim
# from pathlib import Path

# from dlvc.models.segformer_model import DeepSegmenter
# from dlvc.models.segformer import SegFormer
# from trainer_segformer import SegFormerTrainer

# # Replace these with your actual implementations
# from your_dataset import YourSegmentationDataset  # Custom Dataset (implements __getitem__, __len__)
# from your_metrics import IoUMetric                # Metric with .update() and .mIoU()

# def main():
#     # --- Config ---
#     num_classes = 21  # Replace with your actual number of classes
#     batch_size = 4
#     num_epochs = 30
#     learning_rate = 6e-5
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     save_dir = Path("./checkpoints")
#     save_dir.mkdir(parents=True, exist_ok=True)

#     # --- Model ---
#     model = DeepSegmenter(SegFormer(num_classes=num_classes)).to(device)

#     # --- Dataset ---
#     train_dataset = YourSegmentationDataset(split="train")
#     val_dataset = YourSegmentationDataset(split="val")

#     # --- Loss, Optimizer, Scheduler ---
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

#     # --- Metrics ---
#     train_metric = IoUMetric(num_classes=num_classes)
#     val_metric = IoUMetric(num_classes=num_classes)

#     # --- Trainer ---
#     trainer = SegFormerTrainer(
#         model=model,
#         optimizer=optimizer,
#         loss_fn=criterion,
#         lr_scheduler=scheduler,
#         train_metric=train_metric,
#         val_metric=val_metric,
#         train_data=train_dataset,
#         val_data=val_dataset,
#         device=device,
#         num_epochs=num_epochs,
#         training_save_dir=save_dir,
#         batch_size=batch_size,
#         val_frequency=5
#     )

#     # --- Train ---
#     trainer.train()


# if __name__ == "__main__":
#     main()
import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2
from torchvision.transforms.v2 import InterpolationMode

# Debug: confirm Python can find modules
print("[DEBUG] sys.path:", sys.path)

from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.models.segformer import SegFormer
from dlvc.trainer_segformer import SegFormerTrainer

def main():
    # --- Config ---
    num_classes = 3  # background, foreground, boundary
    batch_size = 64
    num_epochs = 30
    learning_rate = 6e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path("./checkpoints_segformer")
    save_dir.mkdir(parents=True, exist_ok=True)
    use_amp = False  # Mixed precision (optional)
    early_stop_patience = 5

    # --- Transforms ---
    img_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(64, 64)),  # Increase if needed
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.long, scale=False),
        v2.Resize(size=(64, 64), interpolation=InterpolationMode.NEAREST),
        v2.Lambda(lambda x: x - 1)
    ])

    # --- Dataset ---
    train_dataset = OxfordPetsCustom(
        root="oxfordpets_dataset",
        split="trainval",
        target_types='segmentation',
        transform=img_transform,
        target_transform=mask_transform,
        download=True
    )

    val_dataset = OxfordPetsCustom(
        root="oxfordpets_dataset",
        split="test",
        target_types='segmentation',
        transform=img_transform,
        target_transform=mask_transform,
        download=True
    )

    # --- Model ---
    model = SegFormer(num_classes=num_classes).to(device)

    # --- Optimizer, Loss, Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # --- Metrics ---
    train_metric = SegMetrics(classes=num_classes)
    val_metric = SegMetrics(classes=num_classes)

    # --- Trainer ---
    trainer = SegFormerTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=criterion,
        lr_scheduler=scheduler,
        train_metric=train_metric,
        val_metric=val_metric,
        train_data=train_dataset,
        val_data=val_dataset,
        device=device,
        num_epochs=num_epochs,
        training_save_dir=save_dir,
        batch_size=batch_size,
        val_frequency=2,
        use_wandb=True  # Set True if logging to W&B
    )

    print("[INFO] Starting training...")
    trainer.train()

    # --- Final Model Save ---
    final_path = save_dir / "final_segformer_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Final model saved to: {final_path}")

if __name__ == "__main__":
    main()
