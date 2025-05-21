import os
import shutil
import random

## Portions of this code were generated with the assistance of ChatGPT (OpenAI, 2025) and subsequently modified by the author.
# [1]OpenAI. (2025). ChatGPT (May 2025 version) [Large language model]. https://chat.openai.com 

input_folder_A = 'C:/Users/ASUS/Desktop/Code/pix2pix&restormer/fundus_dataset_Solomon/A'
input_folder_B =  'C:/Users/ASUS/Desktop/Code/pix2pix&restormer/fundus_dataset_Solomon/B'
output_base = 'C:/Users/ASUS/Desktop/Code/pix2pix&restormer/fundus_dataset_Solomon_v'
train_ratio = 0.7

output_train_A = os.path.join(output_base, 'train/A')
output_train_B = os.path.join(output_base, 'train/B')
output_test_A = os.path.join(output_base, 'test/A')
output_test_B = os.path.join(output_base, 'test/B')

os.makedirs(output_train_A, exist_ok=True)
os.makedirs(output_train_B, exist_ok=True)
os.makedirs(output_test_A, exist_ok=True)
os.makedirs(output_test_B, exist_ok=True)

image_files = [f for f in os.listdir(input_folder_A) if f.endswith(('.png', '.jpg', '.jpeg'))]

random.shuffle(image_files)

split_idx = int(len(image_files) * train_ratio)
train_files = image_files[:split_idx]
test_files = image_files[split_idx:]

print(f" Total images: {len(image_files)}")
print(f" Training: {len(train_files)} images")
print(f" Testing: {len(test_files)} images")

# Copy files
def copy_pair(file_list, src_A, src_B, dst_A, dst_B):
    for img_name in file_list:
        src_path_A = os.path.join(src_A, img_name)
        src_path_B = os.path.join(src_B, img_name)
        if not os.path.exists(src_path_B):
            print(f"Skipping {img_name} (no matching B)")
            continue
        shutil.copy2(src_path_A, os.path.join(dst_A, img_name))
        shutil.copy2(src_path_B, os.path.join(dst_B, img_name))

copy_pair(train_files, input_folder_A, input_folder_B, output_train_A, output_train_B)

copy_pair(test_files, input_folder_A, input_folder_B, output_test_A, output_test_B)

print(" Dataset split complete!")
