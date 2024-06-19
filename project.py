import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation, PillowWriter
import io
import base64
import bcrypt


# Define a function to load and preprocess historical data
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found")
    df = pd.read_csv(file_path)
    closed_prices = df["Close"]
    mm = MinMaxScaler()
    scaled_price = mm.fit_transform(np.array(closed_prices)[..., None]).squeeze()
    return scaled_price, closed_prices, mm

# Define a function to prepare sequences
def prepare_sequences(scaled_price, seq_len):
    X, y = [], []
    for i in range(len(scaled_price) - seq_len):
        X.append(scaled_price[i: i + seq_len])
        y.append(scaled_price[i + seq_len])
    X = np.array(X)[..., None]
    y = np.array(y)[..., None]
    return X, y


# Define the model class
class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1, :])



# Add CSS for background and login page customization
st.markdown(
    """
    <style>
    body {
        background-color: #20232a;
        font-family: 'Roboto', sans-serif;
        margin: 0;
        padding: 0;
    }
    .main {
        background-image: url("https://img.freepik.com/premium-photo/back-view-businessman-looking-virtual-panel-with-graphs-dark-background-businessman-using-holographic-projection-financial-analysis-decision-making-black-background-ai-generated_538213-3790.jpg?size=626&ext=jpg");
        background-size: cover;
        background-attachment: fixed;
        color: #e8eaf6;
        animation: fadeIn 2s ease-in;
        padding: 20px;

    }
    .main h1 {
        background-color:#0000;
        color:#007FFF;
        padding: 10px;
        -webkit-text-stroke: 1px #C0C0C0; /* Red border around text */
         text-stroke: 1px #C0C0C0;

    }
    .main h3 {
        background-color:#0000;
        color:#FF0000;
        padding: 10px;
        -webkit-text-stroke: 1px #FFD700; /* Red border around text */
         text-stroke: 1px #FFD700;

    }

    .login-container {
        text-align: center;
        padding: 50px;
        margin: 10% auto;
        width: 40%;
        background-image: url("https://img.freepik.com/free-photo/light-blue-white-abstract-background_53876-99566.jpg?size=626&ext=jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        animation: slideIn 1s ease-out;
    }
    .login-button {
        background-color: #03a9f4;
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 20px 2px;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .login-button:hover {
        background-color: #0288d1;
        transform: translateY(-3px);
    }
    .stTextInput>div>div>input {
        background-color: #e0f7fa;
        border-radius: 5px;
        padding: 10px;
        color: #00796b;
        border: 1px solid #00796b;
        transition: border-color 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #004d40;
    }
    .stSelectbox>div>div>input {
        background-color: #e0f7fa;
        border-radius: 5px;
        padding: 10px;
        color: #00796b;
        border: 1px solid #00796b;
        transition: border-color 0.3s ease;
    }
    .stSelectbox>div>div>input:focus {
        border-color: #004d40;
    }
    .stButton>button {
        background-color: #03a9f4;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px 2px;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0288d1;
        transform: translateY(-3px);
    }
    .title {
        font-size: 3em;
        text-align: center;
        color: #03a9f4;
        animation: slideDown 1s ease-out;
    }
    .metrics {
        color: #ff9800;
        font-size: 1.5em;
        padding: 10px;
        margin: 10px 0;
        webkit-text-stroke: 1px #FFD700; /* Red border around text */
        text-stroke: 1px #FFD700;
    }
    .extra-style {
        color: #FF5733;
        font-size: 1.5em;
        padding: 10px;
        animation: bounce 2s infinite;
        webkit-text-stroke: 1px #FFD700; /* Red border around text */
        text-stroke: 1px #FFD700;
    }
    .animated-background {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateY(-100%); }
        to { transform: translateY(0); }
    }
    @keyframes slideDown {
        from { transform: translateY(-50px); }
        to { transform: translateY(0); }
    }
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-30px); }
        60% { transform: translateY(-15px); }
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define user credentials
users = {
    "johndoe": bcrypt.hashpw("password1".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
    "janesmith": bcrypt.hashpw("password2".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
    "harish": bcrypt.hashpw("password3".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
    "navthik": bcrypt.hashpw("password4".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
    "karthi": bcrypt.hashpw("password5".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
    "gowtham": bcrypt.hashpw("password6".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
}

# Create login form
st.sidebar.title("Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")


def check_credentials(username, password):
    if username in users and bcrypt.checkpw(password.encode('utf-8'), users[username].encode('utf-8')):
        return True
    return False


if login_button:
    if check_credentials(username, password):
        st.sidebar.success("Login successful")
        st.session_state['authenticated'] = True
    else:
        st.sidebar.error("Invalid username or password")

# Check if user is authenticated
if 'authenticated' in st.session_state and st.session_state['authenticated']:
    # Initialize Streamlit app
    st.title('Stock Price Prediction')

    # List available companies (CSV files) in the "stocks" directory
    stocks_dir = 'stocks'
    available_companies = [f.split('.')[0] for f in os.listdir(stocks_dir) if f.endswith('.csv')]
    selected_company = st.selectbox('Select a company', available_companies)

    # Load the selected company's data
    file_path = os.path.join(stocks_dir, f"{selected_company}.csv")
    scaled_price, closed_prices, mm = load_data(file_path)
    seq_len = 15
    X, y = prepare_sequences(scaled_price, seq_len)

    # Split data into train and test sets
    train_size = int(0.8 * X.shape[0])
    train_x = torch.from_numpy(X[:train_size]).float()
    train_y = torch.from_numpy(y[:train_size]).float()
    test_x = torch.from_numpy(X[train_size:]).float()
    test_y = torch.from_numpy(y[train_size:]).float()

    # Initialize and train the model
    hidden_size = 64
    learning_rate = 0.001
    num_epochs = 200
    model = Model(1, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_x)
        loss = loss_fn(output, train_y)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        output = model(test_x)

    # Inverse transform the predictions and real values
    pred = mm.inverse_transform(output.numpy())
    real = mm.inverse_transform(test_y.numpy())

    # Calculate evaluation metrics
    mae = mean_absolute_error(real, pred)
    mse = mean_squared_error(real, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(real, pred)
    accuracy = r2 * 100  # Convert R-squared to percentage

    # Display the results
    st.subheader(f'{selected_company} Stock Price Prediction')
    st.write(f'<div class="metrics">Mean Absolute Error (MAE): {mae:.4f}</div>', unsafe_allow_html=True)
    st.write(f'<div class="metrics">Mean Squared Error (MSE): {mse:.4f}</div>', unsafe_allow_html=True)
    st.write(f'<div class="metrics">Root Mean Squared Error (RMSE): {rmse:.4f}</div>', unsafe_allow_html=True)
    st.write(f'<div class="metrics">R-squared (R2): {r2:.4f}</div>', unsafe_allow_html=True)
    st.write(f'<div class="metrics">Accuracy (R2): {accuracy:.2f}%</div>', unsafe_allow_html=True)

    # Create a time-lapse video of the stock price prediction
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(real))
    ax.set_ylim(min(real.min(), pred.min()), max(real.max(), pred.max()))
    line_real, = ax.plot([], [], label='Real', color='#0000FF')
    line_pred, = ax.plot([], [], label='Predicted', color='#FF0000')
    ax.legend()


    def init():
        line_real.set_data([], [])
        line_pred.set_data([], [])
        return line_real, line_pred


    def update(frame):
        line_real.set_data(np.arange(frame), real[:frame])
        line_pred.set_data(np.arange(frame), pred[:frame])
        return line_real, line_pred


    ani = FuncAnimation(fig, update, frames=len(real), init_func=init, blit=True)

    # Save animation as a GIF
    gif_path = 'stock_prediction.gif'
    ani.save(gif_path, writer=PillowWriter(fps=10))

    # Display GIF in Streamlit
    with open(gif_path, 'rb') as gif_file:
        gif_bytes = gif_file.read()
        encoded = base64.b64encode(gif_bytes).decode("utf-8")
        gif_html = f'<img src="data:image/gif;base64,{encoded}" alt="Stock Prediction Animation">'
        st.markdown(gif_html, unsafe_allow_html=True)

    # Display the plot as an animation
    st.pyplot(fig)
else:
    st.info("Please log in to access the app.")