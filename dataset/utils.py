import numpy as np
import random
import torch
import torch.nn as nn
import math
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import DiffJPEG, USMSharp, img2tensor
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F

@MODEL_REGISTRY.register()
class DegradationProcessor():

    def __init__(self):
        self.scale = 4
        self.resize_prob = [0.2, 0.7, 0.1]
        self.resize_range = [0.15, 1.5]
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.gray_noise_prob = 0.4
        self.jpeg_range = [30, 95]

        self.second_blur_prob = 0.8
        self.resize_prob2 = [0.3, 0.4, 0.3]
        self.resize_range2 = [0.3, 1.2]
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.gray_noise_prob2 = 0.4
        self.jpeg_range2 = [30, 95]

        self.jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp()  # do usm sharpening

        # ==== kernel settings ====
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size range: 7, 9, ..., 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob = 0.1
        self.blur_kernel_size = 21
        self.blur_sigma = [0.2, 3]
        self.betag_range = [0.5, 4]
        self.betap_range = [1, 2]

        self.blur_kernel_size2 = 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob2 = 0.1
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]

        self.final_sinc_prob = 0.8

        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        # ==== kernel settings ====

        self.select_kernel()

    def select_kernel(self):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        
        self.kernel1 = kernel
        self.kernel2 = kernel2
        self.sinc_kernel = sinc_kernel

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        Each sample in the batch is processed individually with different degradations.
        """
        # training data synthesis
        self.gt = data

        b = self.gt.size(0)
        lq_list = []
        gt_list = []
        gt_usm_list = []

        for i in range(b):
            gt_i = self.gt[i:i+1]
            gt_usm_i = self.usm_sharpener(gt_i)

            kernel1_i = self.kernel1.to(gt_i.device)
            kernel2_i = self.kernel2.to(gt_i.device)
            sinc_kernel_i = self.sinc_kernel.to(gt_i.device)

            ori_h, ori_w = gt_i.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(gt_usm_i, kernel1_i)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.resize_range[1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.resize_range[0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            if np.random.uniform() < self.gaussian_noise_prob:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=self.gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out.to(gt_i.dtype),
                    scale_range=self.poisson_scale_range,
                    gray_prob=self.gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.second_blur_prob:
                out = filter2D(out, kernel2_i)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.resize_range2[1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.resize_range2[0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.scale * scale), int(ori_w / self.scale * scale)), mode=mode)
            # add noise
            if np.random.uniform() < self.gaussian_noise_prob2:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=self.gray_noise_prob2)
            else:
                out = random_add_poisson_noise_pt(
                    out.to(gt_i.dtype),
                    scale_range=self.poisson_scale_range2,
                    gray_prob=self.gray_noise_prob2,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
                out = filter2D(out, sinc_kernel_i)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
                out = filter2D(out, sinc_kernel_i)

            # clamp and round
            lq_i = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # append to list
            lq_list.append(lq_i)
            gt_list.append(gt_i)
            gt_usm_list.append(gt_usm_i)

        # stack the results
        lq = torch.cat(lq_list, dim=0)
        gt = torch.cat(gt_list, dim=0)
        gt_usm = torch.cat(gt_usm_list, dim=0)
        # resize lq to the same size as gt
        lq = F.interpolate(lq, size=(gt.size(2), gt.size(3)), mode='bicubic')
        
        return lq, gt, gt_usm

        # All images are resized to (ori_h // self.scale, ori_w // self.scale)
        
    def to(self, device):
        self.jpeger = self.jpeger.to(device)
        self.usm_sharpener = self.usm_sharpener.to(device)
        
        return self

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DegradationProcessor().to(device)
    # batch_size = 2  # Example batch size
    # gt = torch.rand((batch_size, 3, 1024, 1280), dtype=torch.float32).to(device)
    # lq, gt, gt_usm = model.feed_data(gt)
    # print(f"{lq.shape}\n{gt.shape}")
    # gt = torch.rand((batch_size, 3, 960, 1280), dtype=torch.float32).to(device)
    # lq, gt, gt_usm = model.feed_data(gt)
    # print(f"{lq.shape}\n{gt.shape}")

    from PIL import Image
    from torchvision.transforms.functional import to_pil_image
    img = Image.open('/sensei-fs/users/yyu/mori/ORIGIN.jpg')
    gt = torch.tensor(np.array(img).transpose(2, 0, 1) / 255).unsqueeze(0).float().to(device)
    lq, gt, gt_usm = model.feed_data(gt)
    to_pil_image(lq.squeeze(0).cpu()).save('/sensei-fs/users/yyu/mori/LQ.jpg')
    to_pil_image(gt.squeeze(0).cpu()).save('/sensei-fs/users/yyu/mori/GT.jpg')
    to_pil_image(gt_usm.squeeze(0).cpu()).save('/sensei-fs/users/yyu/mori/GT_USM.jpg')
    print(f"{lq.shape}\n{gt.shape}")