
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

# Portions of this code were generated with the assistance of ChatGPT (OpenAI, 2025) and subsequently modified by the author.
# [1]OpenAI. (2025). ChatGPT (May 2025 version) [Large language model]. https://chat.openai.com 

def crop_border_5pct(img: np.ndarray, final_size: int = 256) -> np.ndarray:
    h, w = img.shape[:2]
    mh, mw = int(0.05 * h), int(0.05 * w)
    trimmed = img[mh:h - mh, mw:w - mw]
    return cv2.resize(trimmed, (final_size, final_size),
                      interpolation=cv2.INTER_AREA)


def match_colour_cast(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
   
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)

    for ch in (1, 2):                                      # a and b only
        s_mean, s_std = src_lab[..., ch].mean(), src_lab[..., ch].std()
        r_mean, r_std = ref_lab[..., ch].mean(), ref_lab[..., ch].std()
        if s_std < 1e-6:
            s_std = 1.0                                    # avoid division by 0
        src_lab[..., ch] = (src_lab[..., ch] - s_mean) / s_std * r_std + r_mean

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


def process_pairs(dir_A: str, dir_B: str, out_A: str, out_B: str,
                  size: int = 256) -> None:

    os.makedirs(out_A, exist_ok=True)
    os.makedirs(out_B, exist_ok=True)

    for pA in tqdm(sorted(glob(os.path.join(dir_A, '*'))), desc='Processing'):
        name = os.path.basename(pA)
        pB = os.path.join(dir_B, name)
        if not os.path.exists(pB):
            tqdm.write(f'  {name}: missing pair in B')
            continue

        imgA = cv2.imread(pA)
        imgB = cv2.imread(pB)
        if imgA is None or imgB is None:
            tqdm.write(f'  {name}: unreadable')
            continue

        cropA = crop_border_5pct(imgA, size)
        cropB = crop_border_5pct(imgB, size)

        cropB_matched = match_colour_cast(cropB, cropA)

        cv2.imwrite(os.path.join(out_A, name), cropA)
        cv2.imwrite(os.path.join(out_B, name), cropB_matched)


if __name__ == '__main__':
    ap = argparse.ArgumentParser('Trim 5 % border and match B colour to A')
    ap.add_argument('--dir_A',
        default=r'C:\Users\ASUS\Desktop\Code\pix2pix&restormer\fundus_dataset_Solomon_v\train\A',
        help='folder with domain-A images')
    ap.add_argument('--dir_B',
        default=r'C:\Users\ASUS\Desktop\Code\pix2pix&restormer\fundus_dataset_Solomon_v\train\B',
        help='folder with domain-B images (paired)')
    ap.add_argument('--out_A',
        default=r'C:\Users\ASUS\Desktop\Code\pix2pix&restormer\fundus_dataset_Solomon_v_crop\train\A',
        help='destination for cropped A images')
    ap.add_argument('--out_B',
        default=r'C:\Users\ASUS\Desktop\Code\pix2pix&restormer\fundus_dataset_Solomon_v_crop\train\B',
        help='destination for cropped B images')
    ap.add_argument('--size', type=int, default=256,
                    help='final square size in pixels')
    args = ap.parse_args()

    process_pairs(args.dir_A, args.dir_B, args.out_A, args.out_B,
                  size=args.size)
