import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import imshow


def dynamic_thresholding(img, perc=99.5):
    s = np.percentile(np.abs(img.ravel()), perc)
    s = np.max([s, 1])
    img = img.clip(-s, s) / s

    return img


class Diffuser:

    def __init__(self, denoiser, class_guidance, diffusion_steps, perc_thresholding=99.5, batch_size=64):
        self.denoiser = denoiser
        self.class_guidance = class_guidance
        self.diffusion_steps = diffusion_steps
        #TODO: parametrize this better:
        self.noise_levels = 1 - np.power(np.arange(0.0001, 0.99, 1 / self.diffusion_steps), 1 / 3)
        self.noise_levels[-1] = 0.01
        self.perc_thresholding = perc_thresholding
        self.batch_size = batch_size

    def predict_x_zero(self, x_t, label, noise_level):
        """Predict original image based on noisy image (or matrix of noisy images) at noise level plus conditional label"""

        # we use 0 for the unconditional embedding:
        num_imgs = len(x_t)
        label_empty_ohe = np.zeros(shape=label.shape)

        # predict x0:
        noise_in = np.array([noise_level] * num_imgs)[:, None, None, None]

        # TODO: can we do some of this in tensorflow?
        # concatenate the conditional and unconditional inputs to speed inference:
        nn_inputs = [np.vstack([x_t, x_t]),
                     np.vstack([noise_in, noise_in]),
                     np.vstack([label, label_empty_ohe])]

        x0_pred = self.denoiser.predict(nn_inputs, batch_size=self.batch_size)

        x0_pred_label = x0_pred[:num_imgs]
        x0_pred_no_label = x0_pred[num_imgs:]

        # classifier free guidance:
        x0_pred = self.class_guidance * x0_pred_label + (1 - self.class_guidance) * x0_pred_no_label

        # clip the prediction using dynamic thresholding a la Imagen:
        x0_pred = dynamic_thresholding(x0_pred, perc=self.perc_thresholding)

        return x0_pred

    def reverse_diffusion(self, seeds, label, show_img=False, masked_imgs=None, mask=None, u=1):
        """Reverse Guided Diffusion on a matrix of random images (seeds). Returns generated images"""

        new_img = seeds

        for i in tqdm(range(len(self.noise_levels) - 1)):

            curr_noise, next_noise = self.noise_levels[i], self.noise_levels[i + 1]

            # predict original denoised image:
            x0_pred = self.predict_x_zero(new_img, label, curr_noise)

            # new image at next_noise level is a weighted average of old image and predicted x0:
            new_img = ((next_noise - curr_noise) * x0_pred + curr_noise * new_img) / next_noise

            if show_img:
                imshow(x0_pred[0])
                plt.show()

        return x0_pred