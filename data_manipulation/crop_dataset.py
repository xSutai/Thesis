
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

# Portions of this code were generated with the assistance of ChatGPT (OpenAI, 2025) and subsequently modified by the author.
# [1]OpenAI. (2025). ChatGPT (May 2025 version) [Large language model]. https://chat.openai.com 

def find_optic_disc_center(img: np.ndarray) -> tuple[int, int]:
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    n, _, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
    if n > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cx, cy = map(int, centroids[idx])
        return cx, cy

    h, w = gray.shape
    return w // 2, h // 2
def crop_at(img: np.ndarray,
            cx: int,
            cy: int,
            size: int = 256) -> np.ndarray:
   
    h, w = img.shape[:2]
    half = size // 2
    cx = int(max(half, min(cx, w - half)))
    cy = int(max(half, min(cy, h - half)))

    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + size, y1 + size
    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    pad_bottom = size - crop.shape[0]
    pad_right  = size - crop.shape[1]

    return cv2.copyMakeBorder(crop,
                              top=0, bottom=pad_bottom,
                              left=0, right=pad_right,
                              borderType=cv2.BORDER_CONSTANT,
                              value=(0, 0, 0))


def process_pairs(dir_A: str,
                  dir_B: str,
                  out_A: str,
                  out_B: str,
                  size: int = 256) -> None:

    os.makedirs(out_A, exist_ok=True)
    os.makedirs(out_B, exist_ok=True)

    for pA in tqdm(sorted(glob(os.path.join(dir_A, '*'))), desc='Cropping'):
        name = os.path.basename(pA)
        pB = os.path.join(dir_B, name)

        if not os.path.exists(pB):
            tqdm.write(f'  {name}: missing pair in B'); continue

        imgA = cv2.imread(pA)
        imgB = cv2.imread(pB)
        if imgA is None or imgB is None:
            tqdm.write(f'  {name}: cannot read'); continue

        cx, cy = find_optic_disc_center(imgA)

        cropA = crop_at(imgA, cx, cy, size)
        cropB = crop_at(imgB, cx, cy, size)

        cv2.imwrite(os.path.join(out_A, name), cropA)
        cv2.imwrite(os.path.join(out_B, name), cropB)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Crop aligned optic-disc patches.')
    p.add_argument('--dir_A',
        default=r'C:\Users\ASUS\Downloads\BrG_test_data\test_bad\all_bad_im',
        help='folder with domain-A images')
    p.add_argument('--dir_B',
        default=r'C:\Users\ASUS\Downloads\BrG_test_data\test_bad\B',
        help='folder with domain-B images (paired, same filenames)')
    p.add_argument('--out_A',
        default=r'C:\Users\ASUS\Downloads\BrG_test_data\test_bad\bad_256\test\A',
        help='destination for cropped A images')
    p.add_argument('--out_B',
        default=r'C:\Users\ASUS\Downloads\BrG_test_data\test_bad\bad_256\test\B',
        help='destination for cropped B images')
    p.add_argument('--size', type=int, default=256,
                   help='crop size in pixels')
    args = p.parse_args()

    process_pairs(args.dir_A, args.dir_B, args.out_A, args.out_B,
                  size=args.size)
