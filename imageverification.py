from pymongo import MongoClient

# MongoDB connection details
mongo_uri = "mongodb+srv://seetarama07:LdrF4mPtz5zdQWsY@imageverification.nywflxg.mongodb.net/?retryWrites=true&w=majority&ssl=true&appName=imageverification"  # Replace with your MongoDB Atlas URI
db_name = "imageverificationdb"  # Your database name in MongoDB Atlas
collection_name = "teams"  # Your collection name

# Connect to MongoDB Atlas
client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]



# Example query
team_name = "example_team"
team = collection.find_one({"team_name": team_name})
print(team)
