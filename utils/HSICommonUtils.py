import torch
import numpy as np
from torchvision import transforms


def ImageStretching(image):
    channels = image.shape[2]
    band_list = []
    for i in range(channels):
        band_data = image[:,:,i]
        band_min = np.min(band_data)
        band_max = np.max(band_data)

        if band_max == band_min:
            # Avoid divide by zero for constant bands
            band_data = np.zeros_like(band_data)
        else:
            band_data = (band_data - band_min) / (band_max - band_min)

        band_list.append(band_data)
    image_data = np.stack(band_list, axis=-1)
    image_data = np.clip(image_data, 0, 1)
    image_data = (image_data * 255).astype(np.uint8)
    image_data = np.uint8(image_data)
    return image_data


def normlize3D(image,use_group=False,group_num=4):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    size = image.shape
    if size[2]!=3:
        image_norms = []

        for i in range(size[2]):
            image_slice3 = image[:,:,i,:,:]
            image_slice_norm = transform(image_slice3)
            image_norms.append(image_slice_norm.unsqueeze(2))
        image_norms = torch.cat(image_norms,dim=2)
        if use_group:
            image_norms = image_norms.unsqueeze(0)
            grouped_channels = []
            for start_channel in range(0,group_num):
                grouped_channels.append(np.arange(start_channel,(image_norms.shape[2]//group_num)*group_num,group_num))
            grouped_img = torch.cat([image_norms[:, :, channels, :, :] for channels in grouped_channels], dim=0)
            return grouped_img.cuda()
        else:
            return image_norms
