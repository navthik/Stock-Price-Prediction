import streamlit as st
from project import load_data, prepare_sequences, Model
from user_management import register_user, check_credentials
import torch
import torch.nn as nn

st.title("Stock Price Prediction App")

# Load and preprocess data
file_path = "C:/Users/gowth/PycharmProjects/pythonProject/stocks/GOOGLE.csv"
scaled_price, closed_prices, mm = load_data(file_path)
seq_len = 15
X, y = prepare_sequences(scaled_price, seq_len)

# Define model and train
hidden_size = 64
learning_rate = 0.001
num_epochs = 200
model = Model(1, hidden_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

st.title("User Authentication Example")

# User registration form
st.subheader("Register")
new_username = st.text_input("Enter a new username", key="new_username")
new_password = st.text_input("Enter a new password", type="password", key="new_password")
if st.button("Register"):
    if register_user(new_username, new_password):
        st.success("Registration successful")
    else:
        st.error("Username already exists")

# User login form
st.subheader("Login")
username = st.text_input("Enter your username", key="username")
password = st.text_input("Enter your password", type="password", key="password")
if st.button("Login"):
    if check_credentials(username, password):
        st.success("Login successful")
    else:
        st.error("Invalid username or password")
