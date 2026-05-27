import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import torch.nn as nn
from PIL import Image
import numpy as np
import os

# ==========================================
# 1. PAGE LAYOUT & FRONTEND STYLING
# ==========================================
st.set_page_config(page_title="NutriVision Dashboard", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 40px; font-weight: bold; color: #FF4B4B; text-align: center; }
    .nutrition-card { background-color: #f9f9f9; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>🍎 NutriVision Full-Stack Scanner</div>", unsafe_allow_html=True)
st.write("Project Working Directory: `D:\\CV_Final_Project`")

# ==========================================
# 2. BACKEND: DEFINE CLASSES & LOAD WEIGHTS
# ==========================================
FOOD101_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding',
    'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate',
    'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich',
    'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari',
    'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyro',
    'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich',
    'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes',
    'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli',
    'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
    'waffles'
]

@st.cache_resource
def load_pytorch_backend():
    model = efficientnet_b0()
    num_classes = len(FOOD101_CLASSES)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    weight_path = r"D:\CV_Final_Project\efficientnet_food101.pth"
    if not os.path.exists(weight_path):
        st.error(f"Missing weight file at: {weight_path}")
        return None
        
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

model = load_pytorch_backend()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. BACKEND: MOCK NUTRITION API
# ==========================================
def fetch_nutrition_data(food_name):
    return {"calories": 320, "carbs": 45, "protein": 12, "fat": 10}

# ==========================================
# 4. FRONTEND CONTENT INTERACTION PIPELINE
# ==========================================
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    
    display_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    
    # --- OPENCV SEGMENTATION (UNTOUCHED VERIFIED VALUES) ---
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_items = {}
    MIN_AREA_SIZE = 15000 
    
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA_SIZE:  
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        crop = display_img[y:y+h, x:x+w]
        
        if crop.size == 0:
            continue
            
        input_tensor = preprocess(crop).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        conf_score = confidence.item() * 100
        predicted_label = FOOD101_CLASSES[predicted_idx.item()]
        
        if conf_score >= 45.0:
            clean_name = predicted_label.replace("_", " ")
            detected_items[clean_name] = conf_score
            
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(display_img, f"{clean_name.upper()} ({conf_score:.1f}%)", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
    # --- UI LAYOUT SECTION ---
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.subheader("📸 Final Output Matrix")
        st.image(display_img, use_container_width=True)
        
    with col2:
        st.subheader("⚙️ X-Ray Mask")
        st.caption("White = Detected Object")
        st.image(thresh, use_container_width=True, clamp=True)
        
    with col3:
        st.subheader("📋 API Nutrition Data")
        if detected_items:
            for item_name, score in detected_items.items():
                st.markdown("<div class='nutrition-card'>", unsafe_allow_html=True)
                st.markdown(f"**Item:** `{item_name.capitalize()}` *(Conf: {score:.1f}%)*")
                
                # FIXED BUG HERE: Changed parameter from clean_name to item_name
                nutrients = fetch_nutrition_data(item_name)
                
                # Now it will reliably loop and load values for every single found item block
                st.write(f"🔥 **Calories:** {nutrients.get('calories')} kcal | 🥩 **Prot:** {nutrients.get('protein')}g")
                st.write(f"🍞 **Carbs:** {nutrients.get('carbs')}g | 🥑 **Fat:** {nutrients.get('fat')}g")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No known items detected with high confidence.")