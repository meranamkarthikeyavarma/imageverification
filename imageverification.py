import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from mtcnn import MTCNN
from pymongo import MongoClient
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# MongoDB connection details
mongo_uri = "mongodb+srv://seetarama07:seetarama07@imageverification.nywflxg.mongodb.net/?retryWrites=true&w=majority&ssl=true&appName=imageverification"
db_name = "imageverificationdb"  # Your database name in MongoDB Atlas
collection_name = "teams"  # Your collection name

# Connect to MongoDB Atlas
try:
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    logger.info("Connected to MongoDB successfully.")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    st.error("Failed to connect to the database. Please check the server logs.")

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
            try:
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
                    logger.warning("No face detected in the first image.")
                    return

                # Detect faces and extract embeddings for the second image
                faces2 = detector.detect_faces(image2)
                if len(faces2) > 0:
                    x2, y2, width2, height2 = faces2[0]['box']
                    cropped_face2 = image2[y2:y2+height2, x2:x2+width2]
                    embeddings2 = DeepFace.represent(cropped_face2, model_name="Facenet")[0]["embedding"]
                else:
                    st.error("No face detected in the second image. Please try again with a different image.")
                    logger.warning("No face detected in the second image.")
                    return

                # Calculate the distance between the embeddings
                distance = np.linalg.norm(np.array(embeddings1) - np.array(embeddings2))
                threshold = 10

                # Compare the distance to the threshold and print the appropriate message
                if distance < threshold:
                    st.success("The two images are of the same person.")
                else:
                    st.error("The two images are of different persons.")
            except Exception as e:
                st.error(f"An error occurred while comparing images: {e}")
                logger.error(f"Error in test_model function: {e}")
        else:
            st.error("Please upload both images before comparing.")

def register():
    st.title('Registration :)')
    with st.form(key='registration_form'):
        team_name = st.text_input('Enter your team name:')
        num_candidates = st.selectbox('Select the number of candidates in the team:', options=[1, 2, 3, 4, 5])
        submit_button = st.form_submit_button(label='Create Team')

    if submit_button:
        try:
            # Check if the team name already exists
            if collection.find_one({team_name: {"$exists": True}}):
                st.error('Team name already available, please select another team name.')
                logger.warning(f"Team name '{team_name}' already exists.")
            else:
                st.session_state.team_name = team_name
                st.session_state.num_candidates = num_candidates
                st.session_state.current_candidate = 0
                st.session_state.candidates = []
                navigate_to_page("add_candidates")
        except Exception as e:
            st.error(f"An error occurred during registration: {e}")
            logger.error(f"Error in register function: {e}")

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
                try:
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

                        # Store the candidate details
                        candidate_info = {
                            "name": person_name,
                            "place": person_place,
                            "embedding": embeddings
                        }

                        # Save candidate info to the database
                        collection.update_one(
                            {"team_name": team_name},
                            {"$push": {"candidates": candidate_info}},
                            upsert=True
                        )
                        
                        st.success(f"Candidate {current_candidate + 1} registered successfully!")
                        st.session_state.current_candidate += 1
                        if st.session_state.current_candidate >= num_candidates:
                            navigate_to_page("verify")
                    else:
                        st.error("No face detected in the uploaded image. Please upload a different image.")
                        logger.warning("No face detected in the uploaded image for candidate.")
                except Exception as e:
                    st.error(f"An error occurred during candidate registration: {e}")
                    logger.error(f"Error in add_candidates function: {e}")
            else:
                st.error("Please upload an image for the candidate.")

def verify():
    st.title("Verify")
    team_name = st.session_state.team_name
    if "verified" not in st.session_state:
        st.session_state.verified = False

    if st.session_state.verified:
        st.write("You are already verified.")
        return

    with st.form(key='verification_form'):
        uploaded_image = st.file_uploader("Upload your image for verification", type=["jpg", "jpeg", "png"])
        submit_button = st.form_submit_button(label='Verify')

    if submit_button:
        if uploaded_image is not None:
            try:
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Create the MTCNN face detector
                detector = MTCNN()
                faces = detector.detect_faces(image)

                if len(faces) > 0:
                    x, y, width, height = faces[0]['box']
                    cropped_face = image[y:y+height, x:x+width]
                    embeddings = DeepFace.represent(cropped_face, model_name="Facenet")[0]["embedding"]

                    # Compare with stored embeddings in the database
                    team_info = collection.find_one({"team_name": team_name})

                    if team_info and "candidates" in team_info:
                        candidates = team_info["candidates"]
                        min_distance = float('inf')
                        matched_candidate = None

                        for candidate in candidates:
                            candidate_embeddings = candidate["embedding"]
                            distance = np.linalg.norm(np.array(embeddings) - np.array(candidate_embeddings))
                            if distance < min_distance:
                                min_distance = distance
                                matched_candidate = candidate

                        if min_distance < 10:
                            st.success(f"Verification successful! You are {matched_candidate['name']} from {matched_candidate['place']}.")
                            st.session_state.verified = True
                        else:
                            st.error("Verification failed. No match found.")
                            logger.warning("Verification failed: No match found.")
                    else:
                        st.error("No candidates found for this team. Please register candidates first.")
                        logger.warning("No candidates found for the team.")
                else:
                    st.error("No face detected in the uploaded image. Please upload a different image.")
                    logger.warning("No face detected in the verification image.")
            except Exception as e:
                st.error(f"An error occurred during verification: {e}")
                logger.error(f"Error in verify function: {e}")
        else:
            st.error("Please upload an image for verification.")

# Page navigation
if st.session_state.current_page == "home":
    home()
elif st.session_state.current_page == "test_model":
    test_model()
elif st.session_state.current_page == "register":
    register()
elif st.session_state.current_page == "add_candidates":
    add_candidates()
elif st.session_state.current_page == "verify":
    verify()
