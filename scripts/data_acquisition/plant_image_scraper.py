# Downloads raw plant images from DuckDuckGo.
# Output: D:/project/new/Poisonous/<species>/ and D:/project/new/Non_Poisonous/<species>/

import os
import time
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageFilter
from duckduckgo_search import DDGS
import numpy as np

# ------------------------------
# CONFIGURATION – TUNED TO YIELD 8–14 IMAGES PER SPECIES AFTER DUPLICATE REMOVAL & BG REMOVAL
# ------------------------------
DESIRED_IMAGES_PER_SPECIES = 20      # Start with 20
MAX_ATTEMPTS = 150                    # Max download attempts per species
MIN_IMAGE_SIZE = 400                  # Minimum width/height in pixels
BLUR_THRESHOLD = 100                  # Lower = more sensitive to blur
FADED_THRESHOLD = 240                 # For very bright/dark images

BASE_DIR = "D:/project/new"           # Main dataset folder
POISONOUS_LIST = "poisonous_species.txt"       # 119 species, one per line
NON_POISONOUS_LIST = "non_poisonous_species.txt" # 120 species

# Blacklisted stock/watermark domains – same as your original
BLACKLISTED_DOMAINS = [
    "pinterest", "dreamstime", "alamy", "shutterstock",
    "amazon", "123rf", "depositphotos", "fineartamerica",
    "istockphoto", "ebay", "etsy"
]

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def is_junk_domain(url):
    """Return True if URL comes from a stock photo site."""
    return any(domain in url.lower() for domain in BLACKLISTED_DOMAINS)

def download_image(url):
    """Download image from URL and return PIL Image object."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except Exception:
        return None
    return None

def is_blurry(img, threshold=BLUR_THRESHOLD):
    """Detect blurry images using variance of edge map."""
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges)
    return np.var(arr) < threshold

def is_faded(img, threshold=FADED_THRESHOLD):
    """Detect extremely bright/dark (faded) images."""
    arr = np.array(img.convert("L"))
    mean = np.mean(arr)
    std = np.std(arr)
    return (mean > threshold and std < 10) or (mean < 20 and std < 10)

def save_image(img, path):
    """Save image in its original format (preserve extension)."""
    try:
        img.save(path)
        return True
    except Exception:
        return False

# ------------------------------
# MAIN SCRAPING FUNCTION
# ------------------------------
def scrape_species(species_list, category):
    """
    species_list : list of species names (strings)
    category     : 'Poisonous' or 'Non_Poisonous'
    """
    base_path = Path(BASE_DIR) / category
    base_path.mkdir(parents=True, exist_ok=True)

    for species in species_list:
        print(f"\n[+] Scraping: {species} ({category})")
        species_dir = base_path / species.replace(" ", "_")
        species_dir.mkdir(exist_ok=True)

        saved_count = 0
        attempt = 0

        with DDGS() as ddgs:
            # Search for plant photos of this species
            for result in ddgs.images(f"{species} plant", max_results=100):
                if saved_count >= DESIRED_IMAGES_PER_SPECIES or attempt >= MAX_ATTEMPTS:
                    break

                url = result.get("image")
                if not url or is_junk_domain(url):
                    continue

                img = download_image(url)
                attempt += 1

                if img is None:
                    continue

                # Quality checks
                if img.width < MIN_IMAGE_SIZE or img.height < MIN_IMAGE_SIZE:
                    continue
                if is_blurry(img) or is_faded(img):
                    continue

                # Determine file extension
                ext = img.format.lower() if img.format else "jpg"
                if ext not in ["jpg", "jpeg", "png"]:
                    ext = "jpg"

                filename = species_dir / f"{saved_count+1:03d}.{ext}"

                if save_image(img, filename):
                    saved_count += 1
                    print(f"    Saved {saved_count:02d}")

        print(f"\tSaved {saved_count} raw images to {species_dir} (will reduce after duplicate.py & remove_bg.py)")
        time.sleep(1)   # Polite delay between species

# ------------------------------
# LOAD SPECIES LISTS
# ------------------------------
def load_species(file_path):
    """Read species names from a text file, one per line."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    # Prepare the text files with exactly 119 and 120 species respectively
    non_poisonous_species = load_species(NON_POISONOUS_LIST)
    poisonous_species = load_species(POISONOUS_LIST)

    print(f"Non‑poisonous species: {len(non_poisonous_species)}")
    print(f"Poisonous species:     {len(poisonous_species)}")

    scrape_species(non_poisonous_species, "Non_Poisonous")
    scrape_species(poisonous_species, "Poisonous")

    print("\n[✓] Scraping complete.")