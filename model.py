import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
        tf.keras.layers.Input(shape=(5,)),  
        tf.keras.layers.Dense(64, activation='relu'),  
        tf.keras.layers.Dense(32, activation='relu'),  
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.3),  
        tf.keras.layers.Dense(1, activation='sigmoid')  
    ])

    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, features, labels):
    """Train the Keras model on the provided features and labels."""
    model.fit(features, labels, epochs=100, batch_size=32)
    return model

def scale_features(features):
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, scaler

def plot_roc_curve(y_true, y_scores):
    """Plot the ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def save_model(model, model_path):
    """Save the Keras model to the specified path."""
    model.save(model_path)

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

    # Print detailed classification report
    print(classification_report(y_test, y_pred))

    # Get predicted probabilities for ROC curve
    y_scores = model.predict(X_test)

    # Plot ROC curve
    plot_roc_curve(y_test, y_scores)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Save the model to the current directory
    save_model(model, 'hbps.keras')
