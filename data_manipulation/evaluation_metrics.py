import os
import cv2
import torch
import lpips
import sewar
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Portions of this code were generated with the assistance of ChatGPT (OpenAI, 2025) and subsequently modified by the author.
# OpenAI. (2025). ChatGPT (May 2025 version) [Large language model]. https://chat.openai.com 

#C:/Users/ASUS/Desktop/Code/pix2pix&restormer/restormer_full_real_ep130_Normal_alligned_dataset_ep100train/restormer_fundus_full_real_b2_alligned_ep100/test_latest/images
#C:/Users/ASUS/Desktop/Code/pix2pix&restormer/pix2pix_cbam_v/test_latest/images
#C:/Users/ASUS/Desktop/Code/restormer 3M/restormer_fundus_nor/test_latest/images
#C:/Users/ASUS/Desktop/Code/pix2pix&restormer/restormer_Simple_GDFNMDTA/restormer_fundus_SimpleRestormer_MDTA_GDFN/test_latest/images
#C:/Users/ASUS/Desktop/Code/restormer 3M/restormer_fundus_nor/test_latest/images
#C:/Users/ASUS/Desktop/Code/pix2pix&restormer/restormer_full_real_ep160_Normal_allig_dataset__solomon/restormer_fundus_full_real_b2_Solomon_allig/test_latest/images
#C:/Users/ASUS/Desktop/Code/restormer_real_b2_Stijn_bad/restormer_fundus_full_real_b2_alligned_ep100/test_latest/images
# === CONFIG ===
img_dir = r"C:/Users/ASUS/Desktop/Code/restormer_real_b2_Solomon_finetun_ep190/restormer_fundus_full_real_b2_alligned_Solomonfine_aug/test_190/images"
loss_fn = lpips.LPIPS(net='alex')  # can use 'vgg'
results = []

real_files = [f for f in os.listdir(img_dir) if f.endswith("_real_B.png")]

for real_name in real_files:
    fake_name = real_name.replace("_real_B", "_fake_B")
    real_path = os.path.join(img_dir, real_name)
    fake_path = os.path.join(img_dir, fake_name)

    real_img = cv2.imread(real_path)
    fake_img = cv2.imread(fake_path)

    if real_img is None or fake_img is None:
        continue

    real_t = torch.from_numpy(real_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    fake_t = torch.from_numpy(fake_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    lpips_score = loss_fn(real_t, fake_t).item()
    ssim_score = ssim(real_img, fake_img, channel_axis=2, data_range=real_img.max() - real_img.min())
    psnr_score = psnr(real_img, fake_img, data_range=real_img.max() - real_img.min())
    vif_score = sewar.full_ref.vifp(real_img, fake_img)
    mae_score = np.mean(np.abs(real_img.astype(np.float32) - fake_img.astype(np.float32)))

    results.append({
        'name': real_name.replace("_real_B", ""),
        'ssim': ssim_score,
        'psnr': psnr_score,
        'lpips': lpips_score,
        'vif': vif_score,
        'mae': mae_score,
    })

if results:
    avg = lambda k: sum([r[k] for r in results]) / len(results)
    print("\n Averages:")
    print(f"SSIM: {avg('ssim'):.4f}")
    print(f"PSNR: {avg('psnr'):.2f} dB")
    print(f"LPIPS: {avg('lpips'):.4f}")
    print(f"VIF: {avg('vif'):.4f}")
    print(f"MAE: {avg('mae'):.2f}")
