import torch
from cnd.ocr.converter import strLabelConverter
from cnd.ocr.model import CRNN
import string
import cv2


class Predictor:
    def __init__(self, model_path, transform, device="cuda"):
        
        alphabet = " "
        alphabet += string.ascii_uppercase
        alphabet += "".join([str(i) for i in range(10)])

        MODEL_PARAMS = {
        "image_height" : 32, 
        "number_input_channels" : 3, 
        "number_class_symbols" : len(alphabet) + 1, 
        "rnn_size" : 64
        }
        state_dict = torch.load(model_path, map_location = 'cpu')['model_state_dict']
        self.model = CRNN(**MODEL_PARAMS)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model = self.model.eval()
        
        self.device = device
        self.converter = strLabelConverter(alphabet)
        self.transform = transform
        
    def preds_converter(self, logits, len_images):
        preds_size = torch.IntTensor([logits.size(0)] * len_images)
        _, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(preds, preds_size, raw=False)
        return sim_preds

    def predict(self, images):
        
        torch_images = []
        for image in images:
        
            if (image.shape[-1] != 3):
                raise Exception("Wrong image format")

            image = self.transform(image).unsqueeze(0).to(self.device)
            torch_images.append(image)
            
         
        input_images = torch.cat(torch_images, dim = 0)
        with torch.no_grad():      
            predict = self.model(input_images)
            text = self.preds_converter(predict, input_images.shape[0])
        
        for i in range(len(text)):
            if text[i].replace(" ", "") == "":
                text[i] = None
        
        return text
