# Import Flask components to build the web app and stream real-time data
from flask import Flask, render_template, Response

# Libraries for data manipulation and loading machine learning models
import pandas as pd
import tensorflow as tf
import json
import time
import os
import logging

# Initialise the Flask application
app = Flask(__name__)

# Suppress unnecessary logs to keep console output clean
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide most TensorFlow log messages
log = logging.getLogger("werkzeug")       # Get Flask's internal logger
log.setLevel(logging.ERROR)               # Only show error messages
tf.get_logger().setLevel("ERROR")         # Suppress TensorFlow logs
tf.autograph.set_verbosity(0)             # Disable AutoGraph debugging output

# Load the pre-trained LSTM model from file
model_path = "../book/lstm_model.h5"
model = tf.keras.models.load_model(model_path)

# Load the dataset
dataset_path = "../data/dataset.parquet"
df = pd.read_parquet(dataset_path, engine="pyarrow")

# Convert 'Timestamp' column to datetime format for indexing and time-based processing
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Drop columns that are not needed for prediction (if they exist)
df = df.drop(columns=['session_counter', 'time_to_failure'], errors='ignore')

# Specify which sensor readings (features) the model expects for prediction
feature_columns = [
    "Flag roping",
    "Platform Position [°]",
    "Platform Motor frequency [HZ]",
    "Temperature platform drive [°C]",
    "Temperature slave drive [°C]",
    "Temperature hoist drive [°C]",
    "Tensione totale film [%]",
    "Current speed cart [%]",
    "Platform motor speed [%]",
    "Lifting motor speed [RPM]",
    "Platform rotation speed [RPM]",
    "Slave rotation speed [M/MIN]",
    "Lifting speed rotation [M/MIN]"
]

# Keep only the last 20% of the dataset for live predictions
split_index = int(len(df) * 0.8)
prediction_data = df.iloc[split_index:].copy()

# Compute mean and standard deviation from the original training data (first 80%)
# Used for normalising the inputs as expected by the pre-trained model
mean = df.iloc[:split_index][feature_columns].mean()
std = df.iloc[:split_index][feature_columns].std()

# Apply normalisation to the prediction data to match training input format
normalised_prediction_data = prediction_data.copy()
normalised_prediction_data[feature_columns] = (prediction_data[feature_columns] - mean) / std


# Function to identify the most deviating sensor readings at the time of prediction
# Helps explain which features likely triggered the alert
def get_top_deviating_features(actual_input, mean, std, top_n=3):
    deviation = ((actual_input - mean) / std).abs()  # Compute z-score deviation
    top_features = deviation.sort_values(ascending=False).head(top_n)  # Pick top deviating features
    result = {}
    for feature in top_features.index:
        result[feature] = {
            "value": float(actual_input[feature]),     # The actual value from the sensor
            "mean": float(mean[feature]),              # Mean from training data
            "z_score": float(deviation[feature])       # Z-score for deviation
        }
    return result


# Generator function that streams prediction results every 5 seconds
# Feeds data to the front-end as Server-Sent Events (SSE)
def generate_predictions():
    window_size = 120  # Input window size for the LSTM (e.g., past 10 minutes)
    shift = 180        # Lookahead window to determine if an alert occurred (e.g., next 15 minutes)
    i = 0              # Start index in the prediction data

    while i + window_size < len(prediction_data):
        # Slice the input window and reshape for the LSTM model
        input_data = normalised_prediction_data.iloc[i: i + window_size][feature_columns].values
        input_data = input_data.reshape(1, 1, window_size * len(feature_columns))  # Flatten time dimension

        # Skip if input shape doesn't match model's expected input
        if input_data.shape[2] != model.input_shape[2]:
            i += 1
            continue

        # Make prediction using the LSTM model
        prob = model.predict(input_data, verbose=0)[0][0]  # Probability output
        predicted_value = 1 if prob >= 0.5 else 0           # Convert to binary prediction

        # Determine actual value from the future 'shift' window
        if i + window_size + shift <= len(prediction_data):
            alert_11_window = prediction_data.iloc[i + window_size: i + window_size + shift]['alert_11']
        else:
            alert_11_window = prediction_data.iloc[i + window_size:]['alert_11']
        actual_value = 1 if alert_11_window.max() == 1 else 0

        # Collect current sensor values and timestamp
        raw_feature_values = prediction_data.iloc[i + window_size][feature_columns].to_dict()
        current_timestamp = prediction_data.index[i + window_size].strftime("%H:%M:%S")
        z_score = abs(prob - actual_value)  # Degree of error in the prediction

        alert_info = None
        # If predicted OR actual shows breakdown, collect alert information
        if predicted_value == 1 or actual_value == 1:
            actual_input = pd.Series(prediction_data.iloc[i + window_size][feature_columns], index=feature_columns)
            influential_features = get_top_deviating_features(actual_input, mean, std)
            alert_info = {
                "time": current_timestamp,
                "z_score": round(z_score, 4),
                "features": influential_features,
                "predicted": int(predicted_value),
                "actual": int(actual_value)
            }

        # Assemble the data payload for front-end use
        data = {
            "time_label": current_timestamp,
            "predicted": int(predicted_value),
            "actual": int(actual_value),
            "z_score": round(z_score, 4),
            "features": raw_feature_values
        }

        # Attach breakdown alert if triggered
        if alert_info:
            data["breakdown"] = alert_info

        # Send the JSON-formatted data as a Server-Sent Event (SSE)
        yield f"data: {json.dumps(data)}\n\n"
        time.sleep(5)  # Pause for 5 seconds before the next prediction
        i += 1         # Slide the prediction window forward

    # Notify the front-end when the dataset has been fully processed
    yield f"data: {json.dumps({'end': True, 'message': 'No more data left to predict.'})}\n\n"


# Define the homepage route — serves the HTML page
@app.route('/')
def index():
    return render_template('index.html')


# Route that streams live prediction data to the front-end
@app.route('/stream')
def stream():
    return Response(generate_predictions(), mimetype='text/event-stream')


# Start the Flask server on port 5001
if __name__ == "__main__":
    port = 5001
    print(f"\n🚀 Server running at: http://127.0.0.1:{port}/\n")
    app.run(port=port, debug=True)
