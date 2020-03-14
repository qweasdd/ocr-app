import torch
from cnd.ocr.converter import strLabelConverter
from cnd.ocr.model import CRNN
import string


class Predictor:
    def __init__(self, model_path, image_size, device="cuda"):
        
        alphabet = " "
        alphabet += string.ascii_uppercase
        alphabet += "".join([str(i) for i in range(10)])

        MODEL_PARAMS = {
        "image_height" : 32, 
        "number_input_channels" : 3, 
        "number_class_symbols" : len(alphabet) + 1, 
        "rnn_size" : 64
        }
        state_dict = torch.load(model_path)['model_state_dict']
        self.model = CRNN(**MODEL_PARAMS)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model = self.model.eval()
        
        self.device = device
        self.ocr_image_size = image_size
        self.converter = strLabelConverter(alphabet)
        
    def preds_converter(self, logits, len_images):
        preds_size = torch.IntTensor([logits.size(0)] * len_images)
        _, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(preds, preds_size, raw=False)
        return sim_preds

    def predict(self, images):
        #TODO: check for correct input type, you can receive one image [x,y,3] or batch [b,x,y,3]
        
        if (images.shape[-1] != 3):
            return None
        
        if (len(images.shape) == 3):
            images = images.unsqueeze(0)
        elif (len(images.shape) != 4):
              return None
            
        images = images.permute(0, 3, 1, 2).to(self.device)
        
        images = images.float() / 255
              
        with torch.no_grad():      
            predict = self.model(images)
            
            text = self.preds_converter(predict, images.shape[0])
        
        return text
