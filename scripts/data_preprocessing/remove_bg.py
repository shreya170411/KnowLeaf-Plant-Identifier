from rembg import remove
from PIL import Image
import os

input_folder = "D:/project/new"
output_folder = "D:/project/segmented"

for root, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.jpg','.jpeg','.png')):
            inp_path = os.path.join(root, file)
            out_path = os.path.join(root.replace(input_folder, output_folder), file)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                img = Image.open(inp_path)
                out_img = remove(img)
                out_path = os.path.splitext(out_path)[0] + ".png"
                out_img.save(out_path)

            except Exception as e:
                print(f"Error on {inp_path}: {e}")
