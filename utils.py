import numpy as np
from PIL import Image


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def save_png(pred_out, save_path, name):

    pred_images = pred_out.squeeze(0).squeeze(0).cpu().detach().numpy()
    pred_images = (pred_images+1)/2*255
    pred_images = pred_images.astype(np.uint8)
    pred_PNG = Image.fromarray(pred_images)
    pred_PNG.convert('L').save(save_path + name + '.png')

    return
