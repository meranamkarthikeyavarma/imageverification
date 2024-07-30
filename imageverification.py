from pymongo import MongoClient, errors

# MongoDB connection details
mongo_uri = "mongodb+srv://seetarama07:LdrF4mPtz5zdQWsY@imageverification.nywflxg.mongodb.net/?retryWrites=true&w=majority&appName=imageverification"
db_name = "imageverificationDB"
collection_name = "teams"

def connect_to_mongodb(uri, db_name, collection_name):
    try:
        # Connect to MongoDB Atlas
        client = MongoClient(uri)
        # Attempt to retrieve server information to verify the connection
        client.server_info()
        print("Connected to MongoDB successfully!")
        
        db = client[db_name]
        collection = db[collection_name]
        return collection
    except errors.ServerSelectionTimeoutError as err:
        print("Failed to connect to MongoDB:", err)
        return None

def get_person_details(collection, team_name, person_id):
    if collection is None:
        print("No connection to the collection.")
        return None
    
    # Query to find the team
    team = collection.find_one({team_name: {"$exists": True}})
    if not team:
        print(f"Team {team_name} not found.")
        return None

    # Access the details of the specific person
    person_details = team[team_name].get(person_id)
    if not person_details:
        print(f"Person {person_id} not found in team {team_name}.")
        return None

    return person_details

# Example usage
collection = connect_to_mongodb(mongo_uri, db_name, collection_name)
team_name = "sagi rama krishnam raju"
person_id = "person1"
person_details = get_person_details(collection, team_name, person_id)

if person_details:
    print(person_details)
