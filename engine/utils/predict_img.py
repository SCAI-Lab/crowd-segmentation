from PIL import Image, ImageOps
from torchvision.transforms import v2
import torch
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../DeepLabV3Plus"))

from dataset.loader import TorchDataset

def predict_img(model, device, input_img_path, predict_img_path):
    model = model.eval()
    #target_img_path = "/home/scai/Documents/img/9_target.png"
    img = Image.open(input_img_path).convert("L")
    #target = Image.open(target_img_path).convert("RGB")
    transform = v2.Compose([
        v2.PILToTensor(),
    ])
    image = transform(img)
    image = image.unsqueeze(0)  # Add additional dimension [1, 256, 256] -> [1, 1, 256, 256]
    # print(image.shape)
    # print(image.unsqueeze(0).shape)
    image = image.to(device, dtype=torch.float32)

    output = model(image)
    pred = output.detach().max(dim=1)[1].cpu().numpy()[0]
    pred = TorchDataset.decode_target(pred).astype(np.uint8)

    pred_img = Image.fromarray(pred)
    pred_img = ImageOps.mirror(pred_img.rotate(-90)) # Not sure why this line is needed but otherwise the image has the wrong orientation

    #img.save("out/image_pred.png")
    #target.save('out/target_pred.png')
    Image.fromarray(pred).save(predict_img_path)

    #print("Test images saved successfully")
    #model = model.train()