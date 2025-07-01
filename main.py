import base64
from PIL import Image
import streamlit as st
from keras.models import load_model
from util import classify, set_background
import requests
import urllib.parse
import sqlite3
import os
import streamlit.components.v1 as components



import gdown

@st.cache_resource
def load_remote_model():
    url = 'https://drive.google.com/file/d1cwXTjJ8KvTlrdqxT5k3tNwf3_HIeSKMU/view?usp=sharing'  
    output = 'pneumonia_model.h5'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return load_model(output)
# Load GoMapsPro API key from secrets
try:
    GOOGLE_API_KEY = st.secrets["general"]["GOOGLE_API_KEY"]
except Exception as e:
    st.error(f"Failed to load API key from secrets.toml: {e}")
    GOOGLE_API_KEY = None  # Fallback; set to None to avoid crashes

# SQLite database setup
DB_FILE = "experiences.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS experiences
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  experience TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Initialize database
if not os.path.exists(DB_FILE):
    init_db()

# Function to load experiences from SQLite
def load_experiences():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, experience FROM experiences ORDER BY timestamp DESC")
    experiences = [{"id": row[0], "name": row[1], "experience": row[2]} for row in c.fetchall()]
    conn.close()
    return experiences

# Function to save an experience
def save_experience(name, experience):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO experiences (name, experience) VALUES (?, ?)", 
              (name if name else "Anonymous", experience))
    conn.commit()
    conn.close()

# Function to delete an experience
def delete_experience(experience_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM experiences WHERE id = ?", (experience_id,))
    conn.commit()
    conn.close()
    st.success("Experience deleted successfully!")
    st.rerun()

# Function to attempt browser-based geolocation (optional)
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
    return None  # JavaScript doesn't return directly to Python

# Sidebar navigation
st.sidebar.title("Navigation")
if st.sidebar.button("Pneumonia Classification"):
    st.session_state.page = "Pneumonia Classification"
if st.sidebar.button("Patient Experiences"):
    st.session_state.page = "Patient Experiences"
if st.sidebar.button("Pneumonia Information"):
    st.session_state.page = "Pneumonia Information"
if st.sidebar.button("Find Nearby Doctors"):
    st.session_state.page = "Find Nearby Doctors"

# Set default page
if 'page' not in st.session_state:
    st.session_state.page = "Pneumonia Classification"

# Page: Pneumonia Classification
if st.session_state.page == "Pneumonia Classification":
    model = load_remote_model()  # Load the specific model file
    class_names = ['PNEUMONIA', 'NORMAL']
    
    st.title('Pneumonia Classification')
    st.header('Please upload a chest X-ray image')
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_container_width=True)
        
        class_name, conf_score = classify(image, model, class_names)
        st.write("## {}".format(class_name))
        st.write("### Confidence: {:.2f}%".format(conf_score * 100))

# Page: Patient Experiences
elif st.session_state.page == "Patient Experiences":
    st.title("Patient Experiences with Pneumonia")
    st.write("Share and read experiences from others who have had pneumonia.")
    
    with st.form(key='experience_form'):
        name = st.text_input("Your Name (optional)")
        experience = st.text_area("Share your experience", height=200)
        submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            if experience.strip():
                save_experience(name, experience)
                st.success("Thank you for sharing your experience!")
                st.rerun()
            else:
                st.error("Please enter your experience before submitting.")

    st.header("Shared Experiences")
    experiences = load_experiences()
    if experiences:
        for exp in experiences:
            with st.expander(f"Experience by {exp['name']}"):
                st.write(exp['experience'])
                if st.button("Delete", key=f"delete_{exp['id']}"):
                    delete_experience(exp['id'])
    else:
        st.write("No experiences shared yet. Be the first to share!")

# Page: Pneumonia Information
elif st.session_state.page == "Pneumonia Information":
    st.title("Pneumonia Information")
    st.header("What is Pneumonia?")
    st.write("""
    Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, 
    causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, 
    viruses, and fungi, can cause pneumonia.
    """)
    st.header("Causes")
    st.write("""
    - **Bacterial**: Streptococcus pneumoniae is the most common cause. Other bacteria include Mycoplasma pneumoniae and Legionella pneumophila.
    - **Viral**: Viruses like influenza, respiratory syncytial virus (RSV), and SARS-CoV-2 (COVID-19) can cause pneumonia.
    - **Fungal**: Common in people with weakened immune systems, caused by fungi like Pneumocystis jirovecii.
    - **Aspiration**: Inhaling food, drink, or vomit into the lungs can lead to aspiration pneumonia.
    """)
    st.header("Symptoms")
    st.write("""
    - Cough (with phlegm or blood)
    - Fever, sweating, and chills
    - Shortness of breath
    - Chest pain when breathing or coughing
    - Fatigue
    - Nausea, vomiting, or diarrhea (especially in children)
    - Confusion (in older adults)
    """)
    st.header("Cures and Treatment")
    st.write("""
    - **Antibiotics**: For bacterial pneumonia, antibiotics like amoxicillin or azithromycin are prescribed.
    - **Antiviral Medications**: For viral pneumonia, drugs like oseltamivir may be used if caused by influenza.
    - **Antifungal Medications**: For fungal pneumonia, antifungal drugs are prescribed.
    - **Supportive Care**: Includes oxygen therapy, fluids, and rest to support recovery.
    - **Hospitalization**: Severe cases may require hospitalization, especially for those with weakened immune systems or complications.
    - **Prevention**: Vaccines (e.g., pneumococcal vaccine, flu vaccine), good hygiene, and avoiding smoking can reduce risk.
    """)
    st.header("When to See a Doctor")
    st.write("""
    Seek medical attention if you experience difficulty breathing, chest pain, persistent fever (102°F or higher), 
    or a persistent cough, especially with pus or blood. Immediate care is crucial for infants, older adults, 
    and those with chronic illnesses.
    """)

# Page: Find Nearby Doctors
elif st.session_state.page == "Find Nearby Doctors":
    st.title("Find Nearby Doctors")
    st.header("Access Your Location")
    st.write("Please enter your location manually (e.g., 'New York, NY'). Browser-based location access is optional.")

    if GOOGLE_API_KEY is None:
        st.error("Cannot proceed: API key is missing. Please configure secrets.toml.")
    else:
        # Manual location input
        manual_location = st.text_input("Enter your city or address (e.g., 'New York, NY')", "")
        if manual_location:
            geocode_url = f"https://maps.gomaps.pro/maps/api/geocode/json?address={urllib.parse.quote(manual_location)}&key={GOOGLE_API_KEY}"
            try:
                geocode_response = requests.get(geocode_url)
                geocode_response.raise_for_status()
                geocode_data = geocode_response.json()
                if geocode_data['status'] == 'OK':
                    lat = geocode_data['results'][0]['geometry']['location']['lat']
                    lng = geocode_data['results'][0]['geometry']['location']['lng']
                    st.session_state.location = {'lat': lat, 'lng': lng}
                    st.write(f"Location set to: ({lat:.4f}, {lng:.4f})")
                else:
                    st.error(f"Geocoding failed: {geocode_data.get('status')} - {geocode_data.get('error_message', 'No details provided')}")
                    st.write(f"Geocode response: {geocode_data}")  # Debug output
            except requests.RequestException as e:
                st.error(f"Error connecting to GoMapsPro Geocoding API: {e}")

        # Try browser-based geolocation (optional)
        if not st.session_state.get('location'):
            get_user_location()

        # Search for doctors if location is available
        if st.session_state.get('location'):
            lat, lng = st.session_state.location['lat'], st.session_state.location['lng']
            # Use keyword to filter for relevant doctors
            places_url = f"https://maps.gomaps.pro/maps/api/place/nearbysearch/json?location={lat},{lng}&radius=5000&type=doctor&keyword=pulmonologist&key={GOOGLE_API_KEY}"
            try:
                places_response = requests.get(places_url)
                places_response.raise_for_status()
                places_data = places_response.json()
                
                if places_data['status'] == 'OK':
                    st.header("Nearby Doctors")
                    doctors = places_data['results'][:5]
                    if doctors:
                        markers = f"color:blue|label:Y|{lat},{lng}"
                        for i, doctor in enumerate(doctors):
                            d_lat = doctor['geometry']['location']['lat']
                            d_lng = doctor['geometry']['location']['lng']
                            markers += f"&markers=color:red|label:{chr(65+i)}|{d_lat},{d_lng}"
                        
                        map_url = f"https://maps.gomaps.pro/maps/api/staticmap?center={lat},{lng}&zoom=13&size=600x300&maptype=roadmap&{markers}&key={GOOGLE_API_KEY}"
                        st.image(map_url, caption="Map showing your location (blue) and nearby doctors (red)")
                        
                        for i, doctor in enumerate(doctors):
                            with st.expander(f"{chr(65+i)}. {doctor['name']}"):
                                st.write(f"**Address**: {doctor.get('vicinity', 'N/A')}")
                                st.write(f"**Rating**: {doctor.get('rating', 'N/A')} ({doctor.get('user_ratings_total', 'N/A')} reviews)")
                                place_id = doctor.get('place_id')
                                if place_id:
                                    details_url = f"https://maps.gomaps.pro/maps/api/place/details/json?place_id={place_id}&fields=formatted_phone_number,website&key={GOOGLE_API_KEY}"
                                    try:
                                        details_response = requests.get(details_url)
                                        details_response.raise_for_status()
                                        details_data = details_response.json()
                                        if details_data['status'] == 'OK':
                                            result = details_data.get('result', {})
                                            st.write(f"**Phone**: {result.get('formatted_phone_number', 'N/A')}")
                                            website = result.get('website')
                                            if website:
                                                st.write(f"**Website**: [{website}]({website})")
                                        else:
                                            st.error(f"Place Details API error: {details_data.get('status')} - {details_data.get('error_message', 'No details provided')}")
                                            st.write(f"Details response: {details_data}")  # Debug output
                                    except requests.RequestException as e:
                                        st.error(f"Error connecting to GoMapsPro Place Details API: {e}")
                    else:
                        st.write("No doctors found within 5km of your location.")
                else:
                    st.error(f"Places API error: {places_data.get('status')} - {places_data.get('error_message', 'No details provided')}")
                    st.write(f"Places response: {places_data}")  # Debug output
            except requests.RequestException as e:
                st.error(f"Error connecting to GoMapsPro Places API: {e}")
