import glob
import os
import numpy as np
import cv2
import argparse
from natsort import natsort
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


class Evaluation:
    def __init__(self, use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'

    def ssim(self, imgA, imgB):
        return ssim(imgA, imgB)

    def psnr(self, imgA, imgB):
        return psnr(imgA, imgB)

    def mse(self, imgA, imgB):
        return mse(imgA, imgB)

    def nmse(self, imgA, imgB):
        mse_score = mse(imgA, imgB)
        variance = np.var(imgA)
        return mse_score / variance if variance != 0 else 0

    def mae(self, imgA, imgB):
        return mae(imgA, imgB)


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def imread(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def format_result(metrics):
    result_str = ""
    for key, (mean, std) in metrics.items():
        result_str += f"{key}: {mean:.4f} ± {std:.4f}, "
    return result_str[:-2]


def evaluate_dirs(dirA, dirB, use_gpu):
    paths_A = fiFindByWildcard(os.path.join(dirA, f'*.{type}'))
    paths_B = fiFindByWildcard(os.path.join(dirB, f'*.{type}'))

    evaluation = Evaluation(use_gpu=use_gpu)

    psnr_results = []
    ssim_results = []
    nmse_results = []
    mse_results = []
    mae_results = []

    for pathA, pathB in zip(paths_A, paths_B):
        imgA = imread(pathA)
        imgB = imread(pathB)

        if imgA is None or imgB is None:
            print(f"Unable to read image: {pathA} or {pathB}")
            continue

        if imgA.shape != imgB.shape:
            print(f"Image size mismatch: {pathA} vs {pathB}")
            continue

        if (imgA == imgB).all():
            ssim_score = evaluation.ssim(imgA, imgB)
            mse_score = evaluation.mse(imgA, imgB)
            nmse_score = evaluation.nmse(imgA, imgB)
            mae_score = evaluation.mae(imgA, imgB)
            print(f"{pathA.split('/')[-1]}, {pathB.split('/')[-1]}, "
                  f"SSIM: {ssim_score:.4f}, MSE: {mse_score:.4f}, "
                  f"NMSE: {nmse_score:.4f}, MAE: {mae_score:.4f}")
            ssim_results.append(ssim_score)
            mse_results.append(mse_score)
            nmse_results.append(nmse_score)
            mae_results.append(mae_score)
        else:
            psnr_score = evaluation.psnr(imgA, imgB)
            ssim_score = evaluation.ssim(imgA, imgB)
            mse_score = evaluation.mse(imgA, imgB)
            nmse_score = evaluation.nmse(imgA, imgB)
            mae_score = evaluation.mae(imgA, imgB)
            print(f"{pathA.split('/')[-1]}, {pathB.split('/')[-1]}, "
                  f"PSNR: {psnr_score:.2f}, SSIM: {ssim_score:.4f}, "
                  f"MSE: {mse_score:.4f}, NMSE: {nmse_score:.4f}, MAE: {mae_score:.4f}")
            psnr_results.append(psnr_score)
            ssim_results.append(ssim_score)
            mse_results.append(mse_score)
            nmse_results.append(nmse_score)
            mae_results.append(mae_score)

    metrics = {
        'PSNR': (np.mean(psnr_results), np.std(psnr_results)) if psnr_results else (0, 0),
        'SSIM': (np.mean(ssim_results), np.std(ssim_results)),
        'MSE': (np.mean(mse_results), np.std(mse_results)),
        'NMSE': (np.mean(nmse_results), np.std(nmse_results)),
        'MAE': (np.mean(mae_results), np.std(mae_results))
    }

    print(f"Final Result: {format_result(metrics)}")
    print(np.max(nmse_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirA', default='/path/to/GT/',
                        type=str, help='directory of ground truth images')
    parser.add_argument('--dirB', default='./output/folder_for_saving_generated_images/',
                        type=str, help='directory of predicted images')
    parser.add_argument('--type', default='png', type=str, help='image file type (e.g., png, jpg)')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='use GPU if available')
    args = parser.parse_args()

    dirA = args.dirA
    dirB = args.dirB
    print(f"Evaluating: {dirB}")
    type = args.type
    use_gpu = args.use_gpu

    if len(dirA) > 0 and len(dirB) > 0:
        evaluate_dirs(dirA, dirB, use_gpu=use_gpu)
