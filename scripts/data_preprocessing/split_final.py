import os
import shutil
import random

input_base = "D:/project/segmented"
train_base = "D:/project/final_train"
val_base = "D:/project/final_val"
split_ratio = 0.8

for category in os.listdir(input_base):
    cat_in = os.path.join(input_base, category)
    all_imgs = []
    # Collect all images from all subfolders
    for root, _, files in os.walk(cat_in):
        for f in files:
            if f.lower().endswith(('.png','.jpg','.jpeg')):
                all_imgs.append(os.path.join(root, f))
    random.shuffle(all_imgs)
    split = int(len(all_imgs) * split_ratio)
    for i, img_path in enumerate(all_imgs):
        target_base = train_base if i < split else val_base
        cat_out = os.path.join(target_base, category)
        os.makedirs(cat_out, exist_ok=True)
        # Add parent folder (species) to filename to avoid overwrite
        parent_folder = os.path.basename(os.path.dirname(img_path))
        base = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        new_name = f"{parent_folder}_{base}{ext}"
        shutil.copy(img_path, os.path.join(cat_out, new_name))

print("Split complete. Check counts again!")
