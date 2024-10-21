import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

# Load CSV data
data_path = 'health_data.csv'  # Ensure this is the correct path to your CSV file

def load_data():
    """Load data from CSV and return features and labels."""
    data = pd.read_csv(data_path)
    X = data[['weight', 'sleepHours', 'bpm', 'systolic', 'diastolic']]  # Features
    y = data['label']  # Labels (0 or 1)
    return X, y

def create_model():
    """Define and return a compiled Keras model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5,)),  # Input layer
        tf.keras.layers.Dense(64, activation='relu'),  # Increased neurons
        tf.keras.layers.Dense(32, activation='relu'),  # Hidden layers
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.3),  # Dropout to prevent overfitting
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer (binary classification)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, features, labels):
    """Train the Keras model on the provided features and labels."""
    model.fit(features, labels, epochs=100, batch_size=32)  # Increased epochs and batch size
    return model

def save_model(model, model_path):
    """Save the Keras model to the specified path."""
    model.save(model_path)

def load_model(model_path):
    """Load a Keras model from the specified path."""
    return tf.keras.models.load_model(model_path)

def predict(model, input_data):
    """Make predictions using the trained model."""
    return model.predict(input_data)

def scale_features(features):
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, scaler

if __name__ == "__main__":
    # Load data and split
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    X_train, scaler = scale_features(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the model
    model = create_model()
    model = train_model(model, X_train, y_train)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")

    # Make predictions and convert to binary format (0 or 1)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Print detailed classification report (Precision, Recall, F1-Score)
    print(classification_report(y_test, y_pred))

    # Save the model to the current directory
    save_model(model, 'saved_model.keras')
