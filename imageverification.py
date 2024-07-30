import streamlit as st
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://seetarama07:LdrF4mPtz5zdQWsY@imageverification.nywflxg.mongodb.net/?retryWrites=true&w=majority&appName=imageverification"

st.write("Before creating client")
client = MongoClient(uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=10000)  # 5000 milliseconds = 5 seconds
st.write("After creating client")

try:
    st.write("Before pinging")
    client.admin.command('ping')
    st.write("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    st.write(e)
    st.write("Exception occurred")
