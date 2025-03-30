from flask import Flask, render_template, Response
import pandas as pd
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

# Split dataset: 80% train, 20% test
split_index = int(len(df) * 0.8)
train_data = df.iloc[:split_index].copy()
test_data = df.iloc[split_index:].copy()

# Calculate mean and std from training data
mean = train_data[feature_columns].mean()
std = train_data[feature_columns].std()

# Normalize test data
normalized_test_data = test_data.copy()
normalized_test_data[feature_columns] = (test_data[feature_columns] - mean) / std

def get_top_deviating_features(actual_input, mean, std, top_n=3):
    deviation = ((actual_input - mean) / std).abs()
    top_features = deviation.sort_values(ascending=False).head(top_n)
    result = {}
    for feature in top_features.index:
        result[feature] = {
            "value": float(actual_input[feature]),
            "mean": float(mean[feature]),
            "z_score": float(deviation[feature])
        }
    return result

def generate_predictions():
    window_size = 120  # 10 minutes
    shift = 180        # 15 minutes
    i = 0

    while i + window_size < len(test_data):
        input_data = normalized_test_data.iloc[i : i + window_size][feature_columns].values
        input_data = input_data.reshape(1, 1, window_size * len(feature_columns))

        if input_data.shape[2] != model.input_shape[2]:
            i += 1
            continue

        prob = model.predict(input_data, verbose=0)[0][0]
        predicted_value = 1 if prob >= 0.5 else 0

        if i + window_size + shift <= len(test_data):
            alert_11_window = test_data.iloc[i + window_size : i + window_size + shift]['alert_11']
        else:
            alert_11_window = test_data.iloc[i + window_size :]['alert_11']

        actual_value = 1 if alert_11_window.max() == 1 else 0

        raw_feature_values = test_data.iloc[i + window_size][feature_columns].to_dict()
        current_timestamp = test_data.index[i + window_size].strftime("%H:%M:%S")
        z_score = abs(prob - actual_value)

        mismatch_info = None
        if predicted_value != actual_value:
            actual_input = pd.Series(test_data.iloc[i + window_size][feature_columns], index=feature_columns)
            influential_features = get_top_deviating_features(actual_input, mean, std)
            mismatch_info = {
                "time": current_timestamp,
                "z_score": round(z_score, 4),
                "features": influential_features
            }

        data = {
            "time_label": current_timestamp,
            "predicted": int(predicted_value),
            "actual": int(actual_value),
            "z_score": round(z_score, 4),
            "features": raw_feature_values
        }

        if mismatch_info:
            data["mismatch"] = mismatch_info

        yield f"data: {json.dumps(data)}\n\n"
        time.sleep(0.5)
        i += 1

    yield f"data: {json.dumps({'end': True, 'message': 'No more data left to predict.'})}\n\n"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    return Response(generate_predictions(), mimetype='text/event-stream')

if __name__ == "__main__":
    port = 5001
    print(f"\n\U0001F680 Server running at: http://127.0.0.1:{port}/\n")
    app.run(port=port, debug=True)