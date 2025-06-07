
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

from dlvc.models.segformer import  SegFormer
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer


def train_finetune(args):

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = train_transform
    val_transform2 = train_transform2


    train_data = OxfordPetsCustom(root="/Users/djem/Desktop/school/DL for VC/ex_2/dlvc/dataset", 
                                split="trainval",
                                target_types='segmentation', 
                                transform=train_transform,
                                target_transform=train_transform2,
                                download=True)

    val_data = OxfordPetsCustom(root="/Users/djem/Desktop/school/DL for VC/ex_2/dlvc/dataset", 
                                split="test",
                                target_types='segmentation', 
                                transform=val_transform,
                                target_transform=val_transform2,
                                download=True)
    

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = DeepSegmenter(SegFormer(num_classes=3))
    
    #Downloading the pre-trained encoder from the cityscapes
    encoder_state_dict = torch.load("saved_models/segformer_city_encoder.pth", map_location=device)
    model.net.encoder.load_state_dict(encoder_state_dict)
    
    # Freeze the encoder
    for param in model.net.encoder.parameters():
        param.requires_grad = False

    # Update optimizer to only train decoder parameters
    optimizer = torch.optim.Adam(model.net.decoder.parameters(), lr=args.lr)


    model.to(device)
    

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255) # remember to ignore label value 255 when training with the Cityscapes datset
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    
    

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    trainer = ImgSemSegTrainer(model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    args.num_epochs, 
                    model_save_dir,
                    batch_size=64,
                    val_frequency = 2)
    trainer.train()
    
    trainer.dispose() 

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser(description='Fine-tuning SegFormer on OxfordPets')
    args.add_argument('--num_epochs', type=int, default=31)
    args.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')


    args = args.parse_args()

    train_finetune(args)