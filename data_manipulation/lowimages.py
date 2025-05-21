import os
import cv2
import numpy as np
from PIL import Image
import zipfile
import random
from io import BytesIO
from tqdm import tqdm  

# Portions of this code were generated with the assistance of ChatGPT (OpenAI, 2025) and subsequently modified by the author.
# [1]OpenAI. (2025). ChatGPT (May 2025 version) [Large language model]. https://chat.openai.com 

#  Set Input and Output ZIP Paths
zip_file_path = "C:/Users/ASUS/Downloads/right_glaucoma.zip"  
output_zip_path = "C:/Users/ASUS/Downloads/lright-glaucoma.zip"  

# Temporary folder for processed images
output_folder = "smartphone_fundus_output"
os.makedirs(output_folder, exist_ok=True)

USE_MOTION_BLUR = True
USE_GAUSSIAN_BLUR = True
USE_LENS_FLARE = True
USE_LOW_RESOLUTION = True
USE_SHADOWING = True
USE_JPEG_COMPRESSION = False  

def crop_to_circle(img):
    h, w = img.shape[:2]
    
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(w, h) // 2
    cv2.circle(mask, center, radius, 255, -1)

    img_circular = cv2.bitwise_and(img, img, mask=mask)

    final_img = cv2.cvtColor(img_circular, cv2.COLOR_BGR2BGRA)  
    final_img[:, :, 3] = mask  

    return final_img

def apply_motion_blur(img, degree=8, angle=45):
    if not USE_MOTION_BLUR:
        return img
    kernel = np.zeros((degree, degree))
    kernel[(degree - 1) // 2, :] = np.ones(degree)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((degree / 2 - 0.5, degree / 2 - 0.5), angle, 1), (degree, degree))
    kernel = kernel / degree
    return cv2.filter2D(img, -1, kernel)

def apply_gaussian_blur(img):
    if not USE_GAUSSIAN_BLUR:
        return img
    return cv2.GaussianBlur(img, (7, 7), 2.5) 

def add_lens_flare(img):
    if not USE_LENS_FLARE:
        return img

    h, w = img.shape[:2]
    flare = np.zeros_like(img, dtype=np.uint8)
 
    center_x, center_y = w // 2, h // 2

    radius = random.randint(30, 50)
    intensity = random.uniform(0.2, 0.4)
    color = (255, 255, 255)

    cv2.circle(flare, (center_x, center_y), radius, color, -1, cv2.LINE_AA)

    return cv2.addWeighted(img, 1, flare, intensity, 0)

def apply_shadowing(img):
    if not USE_SHADOWING:
        return img
    
    h, w, c = img.shape

    shadow_mask = np.zeros((h, w, c), dtype=np.uint8)

    for _ in range(random.randint(2, 4)):  
        x, y = random.randint(0, w//2), random.randint(0, h//2)
        radius = random.randint(min(w, h) // 5, min(w, h) // 3)
        cv2.circle(shadow_mask, (x, y), radius, (random.randint(50, 100),) * c, -1)

    shadow_mask = cv2.GaussianBlur(shadow_mask, (101, 101), 0)

    return cv2.addWeighted(img, 1, shadow_mask, -0.4, 10)

def degrade_image(img):
    
    if USE_LOW_RESOLUTION:
        scale_factor = random.uniform(0.5, 0.7)  
        small = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    img = apply_motion_blur(img, degree=random.randint(5, 15), angle=random.randint(0, 180))
    img = apply_gaussian_blur(img)
    img = crop_to_circle(img)
    img = add_lens_flare(img)
    img = apply_shadowing(img)

    return img

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    image_files = [f for f in zip_ref.namelist() if f.endswith('.png')]

    for image_name in tqdm(image_files, desc="Processing Images"):
        with zip_ref.open(image_name) as file:
            img = Image.open(BytesIO(file.read())).convert("RGB")
            img = np.array(img)  
            degraded_img = degrade_image(img)
            output_path = os.path.join(output_folder, os.path.basename(image_name).replace(".png", ".png"))
            Image.fromarray(degraded_img).save(output_path, "PNG")  

with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in os.listdir(output_folder):
        zipf.write(os.path.join(output_folder, file), file)

for file in os.listdir(output_folder):
    os.remove(os.path.join(output_folder, file))
os.rmdir(output_folder)

print(f" All images cropped to circles & processed with smartphone effects while keeping original colors. Saved in {output_zip_path}.")
