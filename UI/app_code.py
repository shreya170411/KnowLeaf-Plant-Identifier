# app_stream.py
# MUST be first Streamlit command
import streamlit as st
st.set_page_config(page_title="KnowLeaf", layout="centered")

# Other imports
import os, json, numpy as np, tensorflow as tf, cv2
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
from difflib import get_close_matches

# --- Paths ---
DATASET_PATH                 = "D:/project/new"
POISONOUS_PATH               = os.path.join(DATASET_PATH, "Poisonous")
NON_POISONOUS_PATH           = os.path.join(DATASET_PATH, "Non_Poisonous")
MODEL_RESNET_PATH            = "resnet50_best.keras"
MODEL_DENSENET_PATH          = "densenet121_best.keras"
CLASS_LABELS_JSON            = "class_labels.json"         
PLANT_PROFILES_POISONOUS     = "Poisonous_profiles.json"       # poisonous
PLANT_PROFILES_NON_POISONOUS = "Non_poisonous_profiles.json"      # non-poisonous

SIMILARITY_THRESHOLD = 0.6  # Slightly increased threshold

# --- Minimal CSS for look/feel ---
st.markdown("""
<style>
div[data-testid="stFileUploader"]{
  border:1px solid #e6e6e6; padding:1rem 1.25rem; border-radius:14px;
  background:#fafafa; max-width:520px
}
.status-safe{font-weight:800;font-size:30px;color:#1aa251;display:inline-block}
.status-toxic{font-weight:800;font-size:30px;color:#d13b3b;display:inline-block}
.plant-name{font-weight:800;font-size:34px;margin:6px 0 2px 0}
.model-confs{font-size:14px;opacity:.85;margin-bottom:12px}
</style>
""", unsafe_allow_html=True)

# --- Cache models ---
@st.cache_resource
def load_models():
    return (
        tf.keras.models.load_model(MODEL_RESNET_PATH),
        tf.keras.models.load_model(MODEL_DENSENET_PATH),
    )

# --- Load profiles (track source to know which JSON it came from) ---
@st.cache_resource
def load_plant_profiles():
    with open(PLANT_PROFILES_POISONOUS, "r", encoding="utf-8") as f:
        pois_list = json.load(f)
    with open(PLANT_PROFILES_NON_POISONOUS, "r", encoding="utf-8") as f:
        non_list = json.load(f)

    pois = {p["common_name"].lower(): {**p, "source": "poisonous"} for p in pois_list}
    non  = {p["common_name"].lower(): {**p, "source": "non_poisonous"} for p in non_list}
    merged = {**non, **pois}
    return merged

# --- Preprocess ---
def preprocess_image(file):
    img = Image.open(file).convert("RGB").resize((224, 224))
    arr = image.img_to_array(img)/255.0
    return np.expand_dims(arr, 0)

# --- Toxic/Safe ensemble prediction (returns class index + ensemble confidence) ---
def predict_binary(resnet, densenet, arr):
    p1 = resnet.predict(arr, verbose=0)[0]   # probs for [Non_Poisonous, Poisonous]
    p2 = densenet.predict(arr, verbose=0)[0]
    avg = (p1 + p2) / 2.0
    idx = int(np.argmax(avg))
    ensemble_conf = float(avg[idx])
    return idx, ensemble_conf

# --- IMPROVED similarity search with shape features ---
def compare_images(img1, img2):
    try:
        # Convert to different color spaces for better comparison
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        
        # Method 1: HSV Color Histogram (original method)
        hsv1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2HSV)
        
        h1_hsv = cv2.calcHist([hsv1],[0,1],None,[180,256],[0,180,0,256])
        h2_hsv = cv2.calcHist([hsv2],[0,1],None,[180,256],[0,180,0,256])
        cv2.normalize(h1_hsv, h1_hsv, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(h2_hsv, h2_hsv, 0, 1, cv2.NORM_MINMAX)
        color_sim = cv2.compareHist(h1_hsv, h2_hsv, cv2.HISTCMP_CORREL)
        
        # Method 2: Edge-based shape comparison
        gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
        
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        
        # Calculate edge similarity (simple method)
        edge_sim = np.sum(edges1 & edges2) / (np.sum(edges1 | edges2) + 1e-5)
        
        # Method 3: Texture comparison using Sobel
        sobel1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 1, ksize=5)
        sobel2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 1, ksize=5)
        texture_sim = cv2.compareHist(
            cv2.calcHist([sobel1.astype(np.uint8)], [0], None, [256], [0, 256]),
            cv2.calcHist([sobel2.astype(np.uint8)], [0], None, [256], [0, 256]),
            cv2.HISTCMP_CORREL
        )
        
        # Combined similarity (weighted average)
        combined_sim = (color_sim * 0.4 + edge_sim * 0.4 + texture_sim * 0.2)
        
        return max(0, combined_sim)  # Ensure non-negative
    except:
        return -1

def find_best_match(uploaded_img):
    """
    Search BOTH Poisonous and Non_Poisonous folders and return:
    (species_folder_name, is_poisonous_bool), best_similarity
    """
    best_species, best_sim, best_is_poisonous = None, -1, None
    for root, is_pois in [(POISONOUS_PATH, True), (NON_POISONOUS_PATH, False)]:
        if not os.path.isdir(root):
            continue
        for species in os.listdir(root):
            spath = os.path.join(root, species)
            if not os.path.isdir(spath):
                continue
            
            # Check multiple images per species for better accuracy
            similarities = []
            valid_samples = 0
            for fname in os.listdir(spath):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    try:
                        cand = Image.open(os.path.join(spath, fname)).convert("RGB")
                        sim = compare_images(uploaded_img, cand)
                        if sim > 0:  # Only consider valid similarities
                            similarities.append(sim)
                            valid_samples += 1
                            if valid_samples >= 3:  # Check 3 samples max per species
                                break
                    except:
                        pass
            
            if similarities:
                avg_sim = np.mean(similarities)
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_species = species
                    best_is_poisonous = is_pois
    
    return (best_species, best_is_poisonous), best_sim

# --- UI: Detailed Information panel ---
def display_plant_info(profile, is_poisonous, ensemble_conf):
    st.markdown("<div class='status-toxic'>POISONOUS PLANT</div>" if is_poisonous
                else "<div class='status-safe'>SAFE PLANT</div>",
                unsafe_allow_html=True)

    st.markdown(f"<div class='plant-name'>{profile.get('common_name','Unknown').title()}</div>",
                unsafe_allow_html=True)

    st.markdown(
        f"<div class='model-confs'>Ensemble confidence: {ensemble_conf*100:.1f}%</div>",
        unsafe_allow_html=True,
    )

    with st.expander("🌿 Detailed Information", expanded=True):
        sci = profile.get("scientific_name")
        if sci: st.write(f"**Scientific Name:** {sci}")

        if is_poisonous or profile.get("source") == "poisonous":
            for label, key in [
                ("Toxicity", "toxicity"),
                ("Toxic Compounds / Properties", "properties"),
                ("Medicinal Uses (historical / risky)", "medicinal_uses"),
                ("Soil Preference", "soil_preference"),
                ("Precautions / First Aid", "cure_precaution"),
            ]:
                val = profile.get(key)
                if val: st.write(f"**{label}:** {val}")
        else:
            for label, key in [
                ("Edibility", "edibility"),
                ("Properties", "properties"),
                ("Medicinal Uses", "medicinal_uses"),
                ("Soil Preference", "soil_preference"),
                ("Precautions", "cure_precaution"),
            ]:
                val = profile.get(key)
                if val: st.write(f"**{label}:** {val}")

# --- App ---
def main():
    resnet, densenet = load_models()
    plant_db         = load_plant_profiles()

    st.title("🌿KnowLeaf: Smart Plant Identifier")
    st.write("Upload a plant image to identify it and get detailed information")

    src = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if src:
        try:
            img = Image.open(src).convert("RGB")
            col_img, col_info = st.columns(2)

            with col_img:
                st.image(img, caption="Your Plant Image", use_container_width=True)

            with col_info:
                with st.spinner("Analyzing..."):
                    arr = preprocess_image(src)
                    _, ensemble_conf = predict_binary(resnet, densenet, arr)

                with st.spinner("Matching species..."):
                    (best_sp, folder_is_poisonous), sim = find_best_match(img)

                if best_sp and sim >= SIMILARITY_THRESHOLD:
                    key = best_sp.replace("_"," ").lower()
                    profile = plant_db.get(key)
                    if not profile:
                        matches = get_close_matches(key, plant_db.keys(), n=1, cutoff=0.6)
                        if matches:
                            profile = plant_db[matches[0]]

                    if profile:
                        # Toxicity follows the matched folder/profile
                        is_poisonous = folder_is_poisonous or (profile.get("source") == "poisonous")
                        display_plant_info(profile, is_poisonous, ensemble_conf)
                        st.success(f"✅ **Reliable Match Found** (Similarity: {sim:.2f}/1.00)")
                    else:
                        st.warning(f"Found visual match but no profile for: {best_sp.replace('_',' ').title()}")
                        st.info(f"**Visual Similarity:** {sim:.2f}/1.00")
                else:
                    # Clear statement when plant is not in dataset
                    st.error("🔍 **Plant Not Recognized**")
                    
                    if best_sp:
                        st.warning(f"Closest match: **{best_sp.replace('_',' ').title()}** (Similarity: {sim:.2f}/1.00 - Below threshold)")
                        st.info("""
                        **Why this might happen:**
                        - The plant is not in our training dataset
                        - Low similarity score indicates poor match
                        - Image quality or background interference
                        """)
                    else:
                        st.info("""
                        **This plant is completely unknown to our system.**
                        
                        **Safety Recommendations:**
                        ⚠️ Always assume unknown plants may be toxic
                        ⚠️ Do not consume or handle without expert verification
                        ⚠️ Consult botanical experts for accurate identification
                        
                        **For better results, try:**
                        - Clear, well-lit photos showing leaves and flowers
                        - Multiple angles of the same plant
                        - Close-up shots of distinctive features
                        """)

        except UnidentifiedImageError:
            st.error("⚠️ The uploaded file is not a valid image.")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

    st.markdown("---")
    st.caption("Plant identification system | Built with Streamlit | Always verify with experts")

if __name__ == "__main__":
    main()