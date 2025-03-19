from flask import Flask, render_template, Response, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import time
import os
import logging

app = Flask(__name__)

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

# Load the trained LSTM model
model_path = "../book/lstm_model.h5"
model = tf.keras.models.load_model(model_path)

# Load dataset
dataset_path = "../data/dataset.parquet"
df = pd.read_parquet(dataset_path, engine="pyarrow")

# Convert 'Timestamp' to datetime and set it as the index
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Ensure 'alert_11' is a binary integer (0 or 1)
df['alert_11'] = df['alert_11'].apply(lambda x: 1 if x > 0 else 0)

# Drop unnecessary columns
df = df.drop(columns=['session_counter', 'time_to_failure'], errors='ignore')

# Define feature columns
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

# Ensure dataset contains all required features
missing_features = [col for col in feature_columns if col not in df.columns]
if missing_features:
    raise ValueError(f"Missing required features in dataset: {missing_features}")

# Split dataset: first 80% for (already) training, last 20% for predictions
split_index = int(len(df) * 0.8)
train_data = df.iloc[:split_index].copy()
test_data = df.iloc[split_index:].copy()

# Compute mean & std from training data only, for normalization
mean = train_data[feature_columns].mean()
std = train_data[feature_columns].std()

# Normalize test data for LSTM input
normalized_test_data = test_data.copy()
normalized_test_data[feature_columns] = (test_data[feature_columns] - mean) / std

def generate_predictions():
    """
    Generate predictions on the last 20% of data, simulating real-time streaming.
    Predict for 15 minutes => 180 intervals (5s each).
    After sending 180 data points, send a final {"end": true} event.
    """
    window_size = 120  # window width used during training
    shift = 180        # shift used during training (predict next 180 timesteps)
    max_points = 180   # 15 minutes / 5s per step = 180 data points

    start_time = time.time()  # for labeling "current time" on x-axis

    # We iterate up to (len(test_data) - window_size - shift) so we can compare to actual values
    for i in range(len(test_data) - window_size - shift):
        if i >= max_points:
            break

        # Prepare sliding window input
        input_data = normalized_test_data.iloc[i : i + window_size][feature_columns].values
        input_data = input_data.reshape(1, 1, window_size * len(feature_columns))

        # Check feature count matches model input
        expected_feature_count = model.input_shape[2]
        if input_data.shape[2] != expected_feature_count:
            print(f"ERROR: Expected {expected_feature_count} features, got {input_data.shape[2]}")
            continue

        # Predict (probability of alert_11)
        prob = model.predict(input_data, verbose=0)[0][0]
        predicted_value = 1 if prob >= 0.5 else 0

        # Compare with actual value over the shift horizon
        alert_11_window = test_data.iloc[i + window_size : i + window_size + shift]['alert_11']
        actual_value = 1 if alert_11_window.max() == 1 else 0

        # De-normalized (raw) feature values for display
        raw_feature_values = test_data.iloc[i + window_size][feature_columns].to_dict()

        # For the chart's x-axis, we label time as HH:MM:SS
        current_timestamp = start_time + i * 5  # i steps, each 5s
        time_label = time.strftime("%H:%M:%S", time.localtime(current_timestamp))

        data = {
            "time_label": time_label,
            "predicted": int(predicted_value),
            "actual": int(actual_value),
            "features": raw_feature_values
        }
        yield f"data: {json.dumps(data)}\n\n"

        # Sleep 5 seconds to emulate real-time streaming
        time.sleep(0.5)

    # After sending 180 points or exhausting the test range:
    end_data = {"end": True}
    yield f"data: {json.dumps(end_data)}\n\n"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    return Response(generate_predictions(), mimetype='text/event-stream')

if __name__ == "__main__":
    port = 5001
    print(f"\n🚀 Server running at: http://127.0.0.1:{port}/\n")
    app.run(port=port, debug=True)
