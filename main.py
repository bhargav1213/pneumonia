import base64
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from util import classify, set_background
import requests
import urllib.parse
import sqlite3
import os
import streamlit.components.v1 as components
import gdown

MODEL_PATH = "pneumonia_model.h5"

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        file_id = "1cwXTjJ8KvTlrdqxT5k3tNwf3_HIeSKMU"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH, compile=False)

# --- Optional: API key ---
try:
    GOOGLE_API_KEY = st.secrets["general"]["GOOGLE_API_KEY"]
except Exception as e:
    st.error(f"Failed to load API key from secrets.toml: {e}")
    GOOGLE_API_KEY = None

# --- SQLite Setup ---
DB_FILE = "experiences.db"
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    experience TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

if not os.path.exists(DB_FILE):
    init_db()

def load_experiences():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, experience FROM experiences ORDER BY timestamp DESC")
    data = [{"id": row[0], "name": row[1], "experience": row[2]} for row in c.fetchall()]
    conn.close()
    return data

def save_experience(name, experience):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO experiences (name, experience) VALUES (?, ?)", 
              (name if name else "Anonymous", experience))
    conn.commit()
    conn.close()

def delete_experience(experience_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM experiences WHERE id = ?", (experience_id,))
    conn.commit()
    conn.close()
    st.success("Experience deleted successfully!")
    st.rerun()

def get_user_location():
    st.write("Attempting to access your location...")
    geolocation_html = """
    <script>
    function getLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const lat = position.coords.latitude;
                    const lng = position.coords.longitude;
                    window.parent.postMessage({lat: lat, lng: lng}, '*');
                },
                (error) => {
                    window.parent.postMessage({error: error.message}, '*');
                }
            );
        } else {
            window.parent.postMessage({error: 'Geolocation not supported'}, '*');
        }
    }
    getLocation();
    </script>
    """
    components.html(geolocation_html, height=0)
    return None

# --- Sidebar ---
st.sidebar.title("Navigation")
pages = ["Pneumonia Classification", "Patient Experiences", "Pneumonia Information", "Find Nearby Doctors"]
for p in pages:
    if st.sidebar.button(p):
        st.session_state.page = p

if 'page' not in st.session_state:
    st.session_state.page = "Pneumonia Classification"

# --- Page Routing ---
if st.session_state.page == "Pneumonia Classification":
    model = get_model()
    class_names = ['PNEUMONIA', 'NORMAL']
    st.title('Pneumonia Classification')
    st.header('Please upload a chest X-ray image')
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

    if file is not None:
        if model is None:
            st.error("Model not loaded. Cannot perform classification.")
        else:
            image = Image.open(file).convert('RGB')
            st.image(image, use_container_width=True)
            class_name, conf_score = classify(image, model, class_names)
            st.write(f"## {class_name}")
            st.write(f"### Confidence: {conf_score * 100:.2f}%")

elif st.session_state.page == "Patient Experiences":
    st.title("Patient Experiences with Pneumonia")
    with st.form(key='experience_form'):
        name = st.text_input("Your Name (optional)")
        experience = st.text_area("Share your experience", height=200)
        if st.form_submit_button("Submit"):
            if experience.strip():
                save_experience(name, experience)
                st.success("Thank you for sharing!")
                st.rerun()
            else:
                st.error("Experience cannot be empty.")

    for exp in load_experiences():
        with st.expander(f"Experience by {exp['name']}"):
            st.write(exp['experience'])
            if st.button("Delete", key=f"del_{exp['id']}"):
                delete_experience(exp['id'])

elif st.session_state.page == "Pneumonia Information":
    st.title("Pneumonia Information")
    st.markdown("Content about pneumonia causes, symptoms, and treatment goes here...")

elif st.session_state.page == "Find Nearby Doctors":
    st.title("Find Nearby Doctors")
    if GOOGLE_API_KEY is None:
        st.error("API key not found.")
    else:
        location = st.text_input("Enter your city or address")
        if location:
            geocode_url = f"https://maps.gomaps.pro/maps/api/geocode/json?address={urllib.parse.quote(location)}&key={GOOGLE_API_KEY}"
            try:
                r = requests.get(geocode_url).json()
                if r['status'] == 'OK':
                    latlng = r['results'][0]['geometry']['location']
                    st.session_state.location = latlng
                    st.write(f"Location: ({latlng['lat']:.4f}, {latlng['lng']:.4f})")
                else:
                    st.error(f"Geocoding failed: {r['status']}")
            except Exception as e:
                st.error(f"Request failed: {e}")
