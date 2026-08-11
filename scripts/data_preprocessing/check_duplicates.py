import os
from PIL import Image
import imagehash

def remove_duplicates(dataset_path):
    for category in os.listdir(dataset_path):
        cat_path = os.path.join(dataset_path, category)
        hash_dict = {}
        for species in os.listdir(cat_path):
            species_path = os.path.join(cat_path, species)
            for img_name in os.listdir(species_path):
                img_path = os.path.join(species_path, img_name)
                try:
                    img = Image.open(img_path)
                    h = imagehash.phash(img)
                    if h in hash_dict:
                        print("Deleting duplicate:", img_path)
                        os.remove(img_path)
                    else:
                        hash_dict[h] = img_path
                except Exception as e:
                    print("Could not process:", img_path)
remove_duplicates("D:/project/new")
