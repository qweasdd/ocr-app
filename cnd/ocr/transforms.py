import torch
import numpy as np
import albumentations as albu
import cv2

class ResizeToTensor(object):
    def __init__(self, image_size):
        self.image_size = image_size
    
    def __call__(self, image):
        image = image.astype('float32') / 255
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        image = image.astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)

class Transform(object):
    def __init__(self, image_size):
        self.transform = albu.Compose([ albu.ToFloat(p = 1.), albu.Resize(height = image_size[0], width = image_size[1]), albu.RandomBrightnessContrast(p = 0.25), albu.Rotate(limit = 8, p = 0.25)], p = 1.)
        
    def __call__(self, image):
        return torch.from_numpy(self.transform(image = image)['image']).permute(2, 0 , 1)
    
def get_transforms(image_size):
    transform = Transform(image_size)
    return transform
