import streamlit as st
from pymongo import MongoClient, errors

# MongoDB connection details
mongo_uri = "mongodb+srv://seetarama07:LdrF4mPtz5zdQWsY@imageverification.nywflxg.mongodb.net/?retryWrites=true&w=majority&appName=imageverification"
db_name = "imageverificationdb"
collection_name = "teams"

def connect_to_mongodb(uri, db_name, collection_name):
    try:
        # Connect to MongoDB Atlas
        client = MongoClient(uri)
        # Attempt to retrieve server information to verify the connection
        client.server_info()
        st.write("Connected to MongoDB successfully!")
        
        db = client[db_name]
        collection = db[collection_name]
        return collection
    except errors.ServerSelectionTimeoutError as err:
        st.write("Failed to connect to MongoDB:", err)
        return None

def get_person_details(collection, team_name, person_id):
    if collection is None:
        st.write("No connection to the collection.")
        return None
    
    # Query to find the team
    team = collection.find_one({team_name: {"$exists": True}})
    if not team:
        st.write(f"Team {team_name} not found.")
        return None

    # Access the details of the specific person
    person_details = team[team_name].get(person_id)
    if not person_details:
        st.write(f"Person {person_id} not found in team {team_name}.")
        return None

    return person_details

# Streamlit app interface
st.title("MongoDB Data Access")

team_name = st.text_input("Enter team name:")
person_id = st.text_input("Enter person ID:")

if st.button("Submit"):
    collection = connect_to_mongodb(mongo_uri, db_name, collection_name)
    person_details = get_person_details(collection, team_name, person_id)

    if person_details:
        st.write(person_details)
