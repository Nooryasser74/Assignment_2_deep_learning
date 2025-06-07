import os
import torch
import torchvision
import torchvision.transforms.v2 as v2
import os
import matplotlib.pyplot as plt 
import numpy as np

from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.models.segformer import SegFormer
from dlvc.models.segment_model import DeepSegmenter


def imshow(img, filename='img/test.png'):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave(filename,np.transpose(npimg, (1, 2, 0)))
    
def save_mask_grid(masks, filename='mask_grid.png'):
    # Ensure masks have shape [B, H, W]
    if masks.ndim == 3:
        masks = masks  # expected
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    else:
        raise ValueError(f"Unexpected mask shape: {masks.shape}")

    # Expand to [B, 1, H, W] for make_grid
    masks = masks.unsqueeze(1).float() / masks.max()  # normalize to [0, 1]
    grid = torchvision.utils.make_grid(masks, nrow=4)  # shape [3, H, W]
    torchvision.utils.save_image(grid, filename)

    
if __name__ == '__main__': 

    transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    target_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)
                            ])

    dataset = OxfordPetsCustom(root="/Users/djem/Desktop/school/DL for VC/ex_2/dlvc/dataset", 
                            split="test",
                            target_types='segmentation', 
                            transform=transform,
                            target_transform=target_transform,
                            download=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=4,
                                            shuffle=False)

    # Load best model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DeepSegmenter(SegFormer(num_classes=3))
    model.load_state_dict(torch.load("saved_models/segformer_best_finetuned.pth", map_location=device))
    model.eval().to(device)
    
    
    

    # Get one batch
    images, labels = next(iter(dataloader))
    with torch.no_grad():
        preds = model(images.to(device))
        preds = torch.argmax(preds, dim=1)

    # Save images
    torchvision.utils.save_image(images, "img/input_grid.png", nrow=4)
    save_mask_grid(labels, "img/gt_mask_grid.png")
    save_mask_grid(preds, "img/predicted_mask_grid.png")


