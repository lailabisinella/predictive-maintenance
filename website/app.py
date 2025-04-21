# Import Flask components for web app and real-time data streaming
from flask import Flask, render_template, Response

# Data handling and model libraries
import pandas as pd
import tensorflow as tf
import json
import time
import os
import logging

# Initialise Flask app
app = Flask(__name__)

# Suppress TensorFlow and Flask logging to keep the console clean
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

# Load pre-trained LSTM model
model_path = "../book/lstm_model.h5"
model = tf.keras.models.load_model(model_path)

# Load the dataset
dataset_path = "../data/dataset.parquet"
df = pd.read_parquet(dataset_path, engine="pyarrow")

# Convert 'Timestamp' column to datetime format and set as index
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Ensure the target column 'alert_11' is binary (0 or 1)
df['alert_11'] = df['alert_11'].apply(lambda x: 1 if x > 0 else 0)

# Remove unnecessary columns
df = df.drop(columns=['session_counter', 'time_to_failure'], errors='ignore')

# Define the features to be used for model input
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

# Split the dataset: 80% for retraining, 20% for live prediction
split_index = int(len(df) * 0.8)
train_data = df.iloc[:split_index].copy()
prediction_data = df.iloc[split_index:].copy()

# Calculate mean and std for normalisation based on training data
mean = train_data[feature_columns].mean()
std = train_data[feature_columns].std()

# Normalise the prediction data using training stats
normalised_prediction_data = prediction_data.copy()
normalised_prediction_data[feature_columns] = (prediction_data[feature_columns] - mean) / std


# Utility function to get top deviating features using z-score
def get_top_deviating_features(actual_input, mean, std, top_n=3):
    deviation = ((actual_input - mean) / std).abs()  # Absolute z-score
    top_features = deviation.sort_values(ascending=False).head(top_n)
    result = {}
    for feature in top_features.index:
        result[feature] = {
            "value": float(actual_input[feature]),  # Actual value at prediction time
            "mean": float(mean[feature]),  # Mean value (for z-score context)
            "z_score": float(deviation[feature])  # Z-score itself
        }
    return result


# Generator function that yields real-time predictions as server-sent events
def generate_predictions():
    window_size = 120  # 10 minutes reading window
    shift = 180  # 15 minutes prediction window
    i = 0  # Start index

    while i + window_size < len(prediction_data):
        # Extract and reshape input window for LSTM
        input_data = normalised_prediction_data.iloc[i: i + window_size][feature_columns].values
        input_data = input_data.reshape(1, 1, window_size * len(feature_columns))

        # Skip if input shape does not match the model input
        if input_data.shape[2] != model.input_shape[2]:
            i += 1
            continue

        # Predict probability using the LSTM model
        prob = model.predict(input_data, verbose=0)[0][0]
        predicted_value = 1 if prob >= 0.5 else 0  # Convert to binary prediction

        # Define actual value based on next 'shift' window
        if i + window_size + shift <= len(prediction_data):
            alert_11_window = prediction_data.iloc[i + window_size: i + window_size + shift]['alert_11']
        else:
            alert_11_window = prediction_data.iloc[i + window_size:]['alert_11']

        actual_value = 1 if alert_11_window.max() == 1 else 0  # Binary: breakdown occurred or not

        # Get the raw feature values at prediction timestamp
        raw_feature_values = prediction_data.iloc[i + window_size][feature_columns].to_dict()
        current_timestamp = prediction_data.index[i + window_size].strftime("%H:%M:%S")
        z_score = abs(prob - actual_value)  # Deviation of prediction vs Actual

        alert_info = None
        # Trigger alert info if predicted or actual value indicates breakdown
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

        # Prepare data payload for front-end
        data = {
            "time_label": current_timestamp,
            "predicted": int(predicted_value),
            "actual": int(actual_value),
            "z_score": round(z_score, 4),
            "features": raw_feature_values
        }

        # Include alert details if triggered
        if alert_info:
            data["breakdown"] = alert_info

        # Yield data as JSON to be consumed by the web front-end
        yield f"data: {json.dumps(data)}\n\n"
        time.sleep(5)  # 5 seconds pause before next prediction
        i += 1  # Move the window forward

    # Notify client when no more data is available
    yield f"data: {json.dumps({'end': True, 'message': 'No more data left to predict.'})}\n\n"


# Route to render the homepage
@app.route('/')
def index():
    return render_template('index.html')


# Route to stream prediction data to client using Server-Sent Events
@app.route('/stream')
def stream():
    return Response(generate_predictions(), mimetype='text/event-stream')


# Start the Flask web server
if __name__ == "__main__":
    port = 5001
    print(f"\n🚀 Server running at: http://127.0.0.1:{port}/\n")
    app.run(port=port, debug=True)
