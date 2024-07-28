import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from mtcnn import MTCNN
from pymongo import MongoClient

# MongoDB connection details
mongo_uri = "mongodb://localhost:27017/"  # Replace with your MongoDB URI
db_name = "practise"
collection_name = "sample"

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

# Function to manage page navigation
def navigate_to_page(page_name):
    st.session_state.current_page = page_name

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

if "verified" not in st.session_state:
    st.session_state.verified = False

if "captured_embeddings" not in st.session_state:
    st.session_state.captured_embeddings = None

# Define the pages
def home():
    st.markdown("""
        <style>
        .header-container {
            text-align: left;
            margin: 20px;
        }
        .header {
            font-size: 48px;
            font-weight: bold;
            color: #333;
        }
        .caption {
            font-size: 24px;
            color: #666;
            margin-bottom: 20px;
        }
        .button {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background-color: yellow;
            color: black;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 18px;
            font-weight: bold;
            text-decoration: none;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            background-color: #ffd700;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    st.markdown('<div class="header">Are You The Real You?</div>', unsafe_allow_html=True)
    st.markdown('<div class="caption">Let\'s Verify....</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Create vertical spacing using Streamlit's layout features
    st.write("")  # Add empty space to push content down
    st.write("")  # Add more space if needed
    
    # Create three columns for the buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Test the Model"):
            navigate_to_page("test_model")
    with col2:
        if st.button("Register"):
            navigate_to_page("register")
    with col3:
        if st.button("Verify"):
            navigate_to_page("verify")

def test_model():
    st.title("Test the Model")
    st.write("Please upload two photos to compare:")

    # Upload two photos
    uploaded_image1 = st.file_uploader("Upload the first image", type=["jpg", "jpeg", "png"], key="test_image_1")
    uploaded_image2 = st.file_uploader("Upload the second image", type=["jpg", "jpeg", "png"], key="test_image_2")

    if st.button('Compare Images'):
        if uploaded_image1 is not None and uploaded_image2 is not None:
            # Read and process the first image
            file_bytes1 = np.asarray(bytearray(uploaded_image1.read()), dtype=np.uint8)
            image1 = cv2.imdecode(file_bytes1, 1)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

            # Read and process the second image
            file_bytes2 = np.asarray(bytearray(uploaded_image2.read()), dtype=np.uint8)
            image2 = cv2.imdecode(file_bytes2, 1)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

            # Create the MTCNN face detector
            detector = MTCNN()

            # Detect faces and extract embeddings for the first image
            faces1 = detector.detect_faces(image1)
            if len(faces1) > 0:
                x1, y1, width1, height1 = faces1[0]['box']
                cropped_face1 = image1[y1:y1+height1, x1:x1+width1]
                embeddings1 = DeepFace.represent(cropped_face1, model_name="Facenet")[0]["embedding"]
            else:
                st.error("No face detected in the first image. Please try again with a different image.")
                return

            # Detect faces and extract embeddings for the second image
            faces2 = detector.detect_faces(image2)
            if len(faces2) > 0:
                x2, y2, width2, height2 = faces2[0]['box']
                cropped_face2 = image2[y2:y2+height2, x2:x2+width2]
                embeddings2 = DeepFace.represent(cropped_face2, model_name="Facenet")[0]["embedding"]
            else:
                st.error("No face detected in the second image. Please try again with a different image.")
                return

            # Calculate the distance between the embeddings
            distance = np.linalg.norm(np.array(embeddings1) - np.array(embeddings2))
            threshold = 10

            # Compare the distance to the threshold and print the appropriate message
            if distance < threshold:
                st.success("The two images are of the same person.")
            else:
                st.error("The two images are of different persons.")
        else:
            st.error("Please upload both images before comparing.")

def register():
    st.title('Registration :)')
    with st.form(key='registration_form'):
        team_name = st.text_input('Enter your team name:')
        num_candidates = st.selectbox('Select the number of candidates in the team:', options=[1, 2, 3, 4, 5])
        submit_button = st.form_submit_button(label='Create Team')

    if submit_button:
        # Check if the team name already exists
        if collection.find_one({team_name: {"$exists": True}}):
            st.error('Team name already available, please select another team name.')
        else:
            st.session_state.team_name = team_name
            st.session_state.num_candidates = num_candidates
            st.session_state.current_candidate = 0
            st.session_state.candidates = []
            navigate_to_page("add_candidates")

def add_candidates():
    st.title('Add Team Members')
    team_name = st.session_state.team_name
    num_candidates = st.session_state.num_candidates
    current_candidate = st.session_state.current_candidate

    if current_candidate < num_candidates:
        with st.form(key=f'candidate_form_{current_candidate}'):
            st.header(f"Candidate {current_candidate + 1}")
            person_name = st.text_input(f"Person {current_candidate + 1} Name", key=f"person_{current_candidate}_name")
            person_place = st.text_input(f"Person {current_candidate + 1} Place", key=f"person_{current_candidate}_place")
            uploaded_image = st.file_uploader(f"Upload Image for Candidate {current_candidate + 1}", type=["jpg", "jpeg", "png"], key=f"person_{current_candidate}_image")
            submit_button = st.form_submit_button(label='Register Candidate')

        if submit_button:
            if uploaded_image is not None:
                # Read the image and convert it to an appropriate format
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Create the MTCNN face detector
                detector = MTCNN()
                faces = detector.detect_faces(image)

                if len(faces) > 0:
                    # Assuming one face per image for simplicity
                    x, y, width, height = faces[0]['box']
                    cropped_face = image[y:y+height, x:x+width]
                    embeddings = DeepFace.represent(cropped_face, model_name="Facenet")[0]["embedding"]

                    st.session_state.candidates.append({
                        "name": person_name,
                        "place": person_place,
                        "image": embeddings
                    })
                    st.session_state.current_candidate += 1

                    if st.session_state.current_candidate == num_candidates:
                        team_data = {team_name: {f"person{i+1}": st.session_state.candidates[i] for i in range(num_candidates)}}
                        collection.insert_one(team_data)
                        st.success(f'Team "{team_name}" with {num_candidates} candidates has been created!')
                        navigate_to_page("home")
                    else:
                        st.experimental_rerun()
                else:
                    st.error("No face detected in the uploaded image. Please try again with a different image.")
            else:
                st.error("Please upload an image for the candidate.")



def verify():
    st.title("Verify")
    team_name = st.text_input('Enter your team name:')
    person_name = st.text_input('Enter your name:')

    # Initialize person as None
    person = None

    if 'verified' not in st.session_state:
        st.session_state.verified = False

    if st.button('Submit'):
        team = collection.find_one({team_name: {"$exists": True}})
        if not team:
            st.error('Please register first in order to access verify option.')
        else:
            for key, value in team[team_name].items():
                if value['name'] == person_name:
                    person = value
                    break

            if not person:
                st.error('User not found.')
            else:
                st.session_state.verified = True
                st.session_state.person = person  # Store person in session state

    if st.session_state.verified:
        if 'person' not in st.session_state:
            st.error('Unexpected error: person data is not available.')
            return

        person = st.session_state.person

        st.write("Camera feed opened. Click on 'Capture' to take a photo.")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not open camera.")
            return

        ret, frame = cap.read()
        frame_placeholder = st.empty()
        
        if ret:
            frame_placeholder.image(frame, channels="BGR", caption="Camera Feed")
        
        if st.button('Capture', key='capture_button'):
            ret, frame = cap.read()
            if ret:
                captured_image = frame
                st.image(frame, channels="BGR", caption="Captured Image")
                cap.release()
                
                # Process the image to extract embeddings
                image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
                detector = MTCNN()
                faces = detector.detect_faces(image_rgb)

                if faces:
                    x, y, width, height = faces[0]['box']
                    cropped_face = image_rgb[y:y+height, x:x+width]
                    embeddings = DeepFace.represent(cropped_face, model_name="Facenet")[0]["embedding"]

                    if embeddings:
                        st.session_state.captured_embeddings = embeddings
                        # st.success('Embeddings extracted successfully!')

                        # Compare embeddings
                        stored_embeddings = np.array(person['image'])
                        captured_embeddings = np.array(st.session_state.captured_embeddings)
                        distance = np.linalg.norm(stored_embeddings - captured_embeddings)
                        threshold = 10

                        if distance < threshold:
                            st.success('Person verified!')
                        else:
                            st.error('Person not verified.')
                        # st.success(distance)
            
                else:
                    st.error('No face detected. Please try again.')
            else:
                st.error("Error: Failed to capture image.")
                cap.release()


# Main app logic
def main():
    pages = {
        "home": home,
        "register": register,
        "add_candidates": add_candidates,
        "test_model": test_model,
        "verify": verify,
    }

    current_page = st.session_state.current_page
    pages[current_page]()

if __name__ == "__main__":
    main()


print("radhe radhe ")
