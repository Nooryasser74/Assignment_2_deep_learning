
# import argparse
# import os
# import torch
# import torchvision.transforms.v2 as v2
# from pathlib import Path
# from torchvision.models.segmentation import fcn_resnet50

# from dlvc.models.segment_model import DeepSegmenter
# from dlvc.dataset.oxfordpets import  OxfordPetsCustom
# from dlvc.metrics import SegMetrics
# from dlvc.trainer import ImgSemSegTrainer



# def train(args):

#     train_transform = v2.Compose([v2.ToImage(), 
#                             v2.ToDtype(torch.float32, scale=True),
#                             v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
#                             v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
#     train_transform2 = v2.Compose([v2.ToImage(), 
#                             v2.ToDtype(torch.long, scale=False),
#                             v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
#     val_transform = v2.Compose([v2.ToImage(), 
#                             v2.ToDtype(torch.float32, scale=True),
#                             v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
#                             v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
#     val_transform2 = v2.Compose([v2.ToImage(), 
#                             v2.ToDtype(torch.long, scale=False),
#                             v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

#     train_data = OxfordPetsCustom(root="path_to_dataset", 
#                             split="trainval",
#                             target_types='segmentation', 
#                             transform=train_transform,
#                             target_transform=train_transform2,
#                             download=True)

#     val_data = OxfordPetsCustom(root="path_to_dataset", 
#                             split="test",
#                             target_types='segmentation', 
#                             transform=val_transform,
#                             target_transform=val_transform2,
#                             download=True)



#     device = ...

#     model = DeepSegmenter(...)
#     optimizer = ...
#     loss_fn = ...
    
#     train_metric = SegMetrics(classes=train_data.classes_seg)
#     val_metric = SegMetrics(classes=val_data.classes_seg)
#     val_frequency = 2

#     model_save_dir = Path("saved_models")
#     model_save_dir.mkdir(exist_ok=True)

#     lr_scheduler = ...
    
#     trainer = ImgSemSegTrainer(model, 
#                     optimizer,
#                     loss_fn,
#                     lr_scheduler,
#                     train_metric,
#                     val_metric,
#                     train_data,
#                     val_data,
#                     device,
#                     args.num_epochs, 
#                     model_save_dir,
#                     batch_size=64,
#                     val_frequency = val_frequency)
#     trainer.train()

#     # see Reference implementation of ImgSemSegTrainer
#     # just comment if not used
#     trainer.dispose() 

# if __name__ == "__main__":
#     args = argparse.ArgumentParser(description='Training')
#     args.add_argument('-d', '--gpu_id', default='0', type=str,
#                       help='index of which GPU to use')
    
#     if not isinstance(args, tuple):
#         args = args.parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
#     args.gpu_id = 0
#     args.num_epochs = 31


#     train(args)

# import argparse
# import os
# import torch
# import torchvision.transforms.v2 as v2
# from pathlib import Path
# from torchvision.models.segmentation import fcn_resnet50
# import torch.nn as nn

# from dlvc.dataset.oxfordpets import OxfordPetsCustom
# from dlvc.metrics import SegMetrics
# from dlvc.trainer import ImgSemSegTrainer


# def get_fcn_model(num_classes: int, pretrained: bool):
#     print("[DEBUG] Inside get_fcn_model()")
#     model = fcn_resnet50(weights_backbone="DEFAULT" if pretrained else None)
#     print("[DEBUG] FCN model loaded")
#     model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
#     print("[DEBUG] Final layer modified")
#     return model


# def train(args, use_pretrained=True):
#     print("[DEBUG] train() started")
#     print(f"[DEBUG] use_pretrained = {use_pretrained}")

#     print("[DEBUG] Creating transforms...")
#     img_transform = v2.Compose([
#         v2.ToImage(),
#         v2.ToDtype(torch.float32, scale=True),
#         v2.Resize(size=(64, 64)),
#         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     print("[DEBUG] Image transform done")

#     target_transform = v2.Compose([
#         v2.ToImage(),
#         v2.ToDtype(torch.long, scale=False),
#         v2.Resize(size=(64, 64))
#     ])
#     print("[DEBUG] Target transform done")

#     print("[DEBUG] Loading training dataset...")
#     train_data = OxfordPetsCustom(
#         root="oxfordpets_dataset",
#         split="trainval",
#         target_types='segmentation',
#         transform=img_transform,
#         target_transform=target_transform,
#         download=True
#     )
#     print("[DEBUG] Training dataset loaded")

#     print("[DEBUG] Loading validation dataset...")
#     val_data = OxfordPetsCustom(
#         root="oxfordpets_dataset",
#         split="test",
#         target_types='segmentation',
#         transform=img_transform,
#         target_transform=target_transform,
#         download=True
#     )
#     print("[DEBUG] Validation dataset loaded")

#     num_classes = train_data.classes_seg
#     print(f"[DEBUG] Number of classes: {num_classes}")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[DEBUG] Using device: {device}")

#     print("[DEBUG] Getting model...")
#     model = get_fcn_model(num_classes=num_classes, pretrained=use_pretrained)
#     model = model.to(device)
#     print("[DEBUG] Model prepared and moved to device")

#     print("[DEBUG] Initializing optimizer, loss function, and scheduler...")
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
#     lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
#     loss_fn = nn.CrossEntropyLoss(ignore_index=255)
#     print("[DEBUG] Optimizer, loss, and scheduler initialized")

#     print("[DEBUG] Creating metric trackers...")
#     train_metric = SegMetrics(classes=num_classes)
#     val_metric = SegMetrics(classes=num_classes)
#     print("[DEBUG] Metric trackers created")

#     model_dir = "saved_fcn_pretrained" if use_pretrained else "saved_fcn_scratch"
#     Path(model_dir).mkdir(exist_ok=True)
#     print(f"[DEBUG] Model directory: {model_dir}")

#     print("[DEBUG] Initializing trainer...")
#     trainer = ImgSemSegTrainer(
#         model=model,
#         optimizer=optimizer,
#         loss_fn=loss_fn,
#         lr_scheduler=lr_scheduler,
#         train_metric=train_metric,
#         val_metric=val_metric,
#         train_data=train_data,
#         val_data=val_data,
#         device=device,
#         num_epochs=args.num_epochs,
#         training_save_dir=Path(model_dir),
#         batch_size=64,
#         val_frequency=2
#     )
#     print("[DEBUG] Trainer initialized")

#     print("[DEBUG] Starting training loop...")
#     trainer.train()
#     print("[DEBUG] Training complete")

#     trainer.dispose()
#     print("[DEBUG] Trainer disposed")


# if __name__ == "__main__":
#     print("[DEBUG] __main__ block started")

#     parser = argparse.ArgumentParser(description='Train FCN model')
#     parser.add_argument('--gpu_id', default='0', type=str, help='GPU id')
#     parser.add_argument('--num_epochs', default=30, type=int)
#     parser.add_argument('--mode', type=str, choices=["pretrained", "scratch"], required=True)
#     args = parser.parse_args()
#     print(f"[DEBUG] Parsed args: {args}")

#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
#     print(f"[DEBUG] Set CUDA_VISIBLE_DEVICES to {args.gpu_id}")

#     print(f"[DEBUG] Starting training with mode: {args.mode}")
#     train(args, use_pretrained=(args.mode == "pretrained"))
#     print("[DEBUG] Script finished.")
print("[DEBUG] Starting script...")

import argparse
print("[DEBUG] argparse imported")

import os
print("[DEBUG] os imported")

import torch
print("[DEBUG] torch imported")

import torchvision.transforms.v2 as v2
print("[DEBUG] torchvision.transforms.v2 imported")

from torchvision.transforms.v2 import InterpolationMode
print("[DEBUG] InterpolationMode imported")

from pathlib import Path
print("[DEBUG] Path imported from pathlib")

from torchvision.models.segmentation import fcn_resnet50
print("[DEBUG] fcn_resnet50 imported")

import torch.nn as nn
print("[DEBUG] torch.nn imported")

from dlvc.dataset.oxfordpets import OxfordPetsCustom
print("[DEBUG] OxfordPetsCustom imported")

from dlvc.metrics import SegMetrics
print("[DEBUG] SegMetrics imported")

from dlvc.trainer import ImgSemSegTrainer
print("[DEBUG] ImgSemSegTrainer imported")


def get_fcn_model(num_classes: int, pretrained: bool):
    print("[DEBUG] Inside get_fcn_model()")
    model = fcn_resnet50(weights_backbone="DEFAULT" if pretrained else None)
    print("[DEBUG] FCN model loaded")
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    print("[DEBUG] Final layer modified")
    return model


def train(args, use_pretrained=True):
    print("[DEBUG] train() started")
    print(f"[DEBUG] use_pretrained = {use_pretrained}")

    # Image transformations
    print("[DEBUG] Creating transforms...")
    img_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(64, 64)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("[DEBUG] Image transform created")

    # ✅ FIX: Use NEAREST interpolation for label masks
    target_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.long, scale=False),
        v2.Resize(size=(64, 64), interpolation=InterpolationMode.NEAREST),
        v2.Lambda(lambda x: x - 1)
    ])
    print("[DEBUG] Target transform created")

    # Datasets
    print("[DEBUG] Loading training dataset...")
    train_data = OxfordPetsCustom(
        root="oxfordpets_dataset",
        split="trainval",
        target_types='segmentation',
        transform=img_transform,
        target_transform=target_transform,
        download=True
    )
    print("[DEBUG] Training dataset loaded")

    print("[DEBUG] Loading validation dataset...")
    val_data = OxfordPetsCustom(
        root="oxfordpets_dataset",
        split="test",
        target_types='segmentation',
        transform=img_transform,
        target_transform=target_transform,
        download=True
    )
    print("[DEBUG] Validation dataset loaded")

    # ✅ FIXED: Ensure class count is based on class IDs
    class_ids = [c.id for c in train_data.classes_seg]
    num_classes = max(class_ids) + 1
    print(f"[DEBUG] Computed class IDs: {class_ids}")
    print(f"[DEBUG] Final computed num_classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")

    # Model setup
    print("[DEBUG] Getting model...")
    model = get_fcn_model(num_classes=num_classes, pretrained=use_pretrained)
    model = model.to(device)
    print("[DEBUG] Model ready and moved to device")

    # Training tools
    print("[DEBUG] Initializing optimizer, loss, and scheduler...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    print("[DEBUG] Optimizer, LR scheduler, and loss function ready")

    print("[DEBUG] Setting up metrics...")
    train_metric = SegMetrics(classes=num_classes)
    val_metric = SegMetrics(classes=num_classes)
    print("[DEBUG] Metrics ready")

    # Save directory
    model_dir = "saved_fcn_pretrained" if use_pretrained else "saved_fcn_scratch"
    Path(model_dir).mkdir(exist_ok=True)
    print(f"[DEBUG] Model will be saved to: {model_dir}")

    # Trainer
    print("[DEBUG] Creating trainer...")
    trainer = ImgSemSegTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=lr_scheduler,
        train_metric=train_metric,
        val_metric=val_metric,
        train_data=train_data,
        val_data=val_data,
        device=device,
        num_epochs=args.num_epochs,
        training_save_dir=Path(model_dir),
        batch_size=64,
        val_frequency=2
    )
    print("[DEBUG] Trainer created successfully")

    print("[DEBUG] Starting training...")
    trainer.train()
    print("[DEBUG] Training complete")


if __name__ == "__main__":
    print("[DEBUG] Entered __main__")
    parser = argparse.ArgumentParser(description='Train FCN model')
    parser.add_argument('--gpu_id', default='0', type=str, help='GPU id')
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--mode', type=str, choices=["pretrained", "scratch"], required=True)
    args = parser.parse_args()
    print(f"[DEBUG] Parsed args: {args}")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print(f"[DEBUG] CUDA_VISIBLE_DEVICES set to {args.gpu_id}")

    print("[DEBUG] Calling train()...")
    train(args, use_pretrained=(args.mode == "pretrained"))
