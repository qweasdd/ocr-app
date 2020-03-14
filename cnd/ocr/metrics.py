import torch.nn as nn
from cnd.ocr.converter import strLabelConverter
from catalyst.utils.torch import any2device
import torch

class WrapCTCLoss(nn.Module):
    def __init__(self, alphabet, device='cpu'):
        super().__init__()
        self.converter = strLabelConverter(alphabet)
        self.device = device
        self.loss = nn.CTCLoss()

    def preds_converter(self, logits, len_images):
        preds_size = torch.IntTensor([logits.size(0)] * len_images)
        _, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(preds, preds_size, raw=False)
        return sim_preds, preds_size

    def __call__(self, logits, targets):
        temp = self.converter.encode(targets)
        text, length = temp[0].to(self.device), temp[1].to(self.device)
        sim_preds, preds_size = self.preds_converter(logits, len(targets))
        loss = self.loss(logits, text, preds_size, length)
        return loss


#TODO: ADD ACCURACY https://catalyst-team.github.io/catalyst/_modules/catalyst/dl/callbacks/metrics/accuracy.html
# YOU WILL NEED TO WRAP STANDARD ACCURACY, AS CTCLOSS ABOVE
# https://github.com/catalyst-team/catalyst-info#catalyst-info-5-callbacks
