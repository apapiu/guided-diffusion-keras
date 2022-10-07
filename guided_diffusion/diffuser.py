import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import imshow, plot_images


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

        if self.perc_thresholding:
            # clip the prediction using dynamic thresholding a la Imagen:
            x0_pred = dynamic_thresholding(x0_pred, perc=self.perc_thresholding)

        return x0_pred

    def reverse_diffusion(self, seeds, label, show_img=False):
        """Reverse Guided Diffusion on a matrix of random images (seeds). Returns generated images"""

        new_img = seeds

        for i in tqdm(range(len(self.noise_levels) - 1)):

            curr_noise, next_noise = self.noise_levels[i], self.noise_levels[i + 1]

            # predict original denoised image:
            x0_pred = self.predict_x_zero(new_img, label, curr_noise)

            # new image at next_noise level is a weighted average of old image and predicted x0:
            new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise

            if show_img:
                plot_images(x0_pred, nrows=np.sqrt(len(new_img)),
                            save_name=str(i),
                            size=12)
                plt.show()

        return x0_pred

    def reverse_diffusion_masked(self, seeds, label_ohe, show_img=False, masked_imgs=None, mask=None, u=1):
        """Reverse Guided Diffusion on a matrix of random images (seeds) with a mask. Can be used for in/outpainting
        Based on the algorithm from Repaint: https://github.com/andreas128/RePaint
        """

        new_img = seeds
        num_imgs = len(new_img)

        for i in tqdm(range(len(self.noise_levels) - 1)):

            curr_noise, next_noise = self.noise_levels[i], self.noise_levels[i + 1]  # alpha1, alpha2

            for j in range(1, u + 1):

                # predict original denoised image:
                x0_pred = self.predict_x_zero(new_img, label_ohe, curr_noise)

                # new image at next_noise level is a weighted average of old image and predicted x0:
                new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise

                if masked_imgs is not None:
                    # let the last 20% of steps be done without masking.
                    if i <= int(self.diffusion_steps * 0.8):
                        ####masked part:
                        new_img_known = (1 - next_noise) * masked_imgs + np.random.normal(0, 1,
                                                                                          size=new_img.shape) * next_noise

                        # mask here is empty/unknown part
                        new_img = new_img_known * mask + new_img * (1 - mask)
                    else:
                        break

                if j != u:
                    print(j)
                    ### noise the image back to the previous step:
                    s = (1 - curr_noise) / (1 - next_noise)
                    new_img = s * new_img + np.sqrt(curr_noise ** 2 - s ** 2 * next_noise ** 2) * np.random.normal(0, 1,
                                                                                                                   size=new_img.shape)

            if show_img:
                plot_images(x0_pred, nrows=np.sqrt(len(new_img)),
                            save_name=str(i),
                            size=12)
                plt.show()

        return x0_pred