###################
# REVERSE DIFFUSION:
###################

import numpy as np
from tqdm import tqdm


def dynamic_thresholding(img, perc=98):
    s = np.percentile(np.abs(img.ravel()), perc)
    s = np.max([s, 1])
    img = img.clip(-s, s) / s

    return img


class Diffuser():

    def __init__(self, denoiser, class_guidance, diffusion_steps):
        self.denoiser = denoiser
        self.class_guidance = class_guidance
        self.diffusion_steps = diffusion_steps
        self.noise_levels = 1 - np.power(np.arange(0.0001, 0.99, 1 / self.diffusion_steps), 1 / 3)
        self.noise_levels[-1] = 0.01

    def reverse_diffusion(self, seeds, label_ohe, show_img=False, batch_size=64):
        """Reverse Guided Diffusion on a matrix of random images (seeds). Returns generated images"""

        new_img = seeds
        num_imgs = len(seeds)

        # we use 0 for the unconditional embedding:
        label_empty_ohe = np.zeros(shape=label_ohe.shape)

        for noise in tqdm(self.noise_levels):

            if noise != self.noise_levels[0]:
                # new image at alpha is a weighted average of old image and predicted x0
                new_img = ((old_noise - noise) * x0_pred + noise * new_img) / old_noise

            noise_in = np.array([noise] * num_imgs)[:, None, None, None]

            # TODO: can we do some of this in tensorflow directly?
            # concatenate the conditional and unconditional inputs to speed inference:
            nn_inputs = [np.vstack([new_img, new_img]),
                         np.vstack([noise_in, noise_in]),
                         np.vstack([label_ohe, label_empty_ohe])]

            x0_pred = self.denoiser.predict(nn_inputs, batch_size=batch_size)

            x0_pred_label = x0_pred[:num_imgs]
            x0_pred_no_label = x0_pred[num_imgs:]

            # classifier free guidance:
            x0_pred = self.class_guidance * x0_pred_label + (1 - self.class_guidance) * x0_pred_no_label

            # clip the prediction using dynamic thresholding a la Imagen:
            x0_pred = dynamic_thresholding(x0_pred, perc=99)

            old_noise = noise

            if show_img:
                print(noise)
                imshow(x0_pred[0])
                plt.show()

        return new_img