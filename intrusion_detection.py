import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Step 1: Downloading NSL-KDD Dataset...")
    # Using direct raw GitHub links to fetch the NSL-KDD dataset
    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
    test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

    # Feature names for NSL-KDD
    col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
                 "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
                 "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
                 "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
                 "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
                 "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
                 "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
                 "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "difficulty"]

    train_df = pd.read_csv(train_url, header=None, names=col_names)
    test_df = pd.read_csv(test_url, header=None, names=col_names)

    # Drop the difficulty column as it's not needed for classification
    train_df.drop('difficulty', axis=1, inplace=True)
    test_df.drop('difficulty', axis=1, inplace=True)

    print("Step 2: Preprocessing Data (Encoding & Scaling)...")
    # Convert multi-class labels to Binary (Normal = 0 vs Attack = 1)
    train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Combine train and test temporarily for consistent categorical encoding
    combined_df = pd.concat([train_df, test_df])

    # Label Encoding for categorical features (protocol_type, service, flag)
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col])

    # Split back to train and test
    train_df = combined_df.iloc[:len(train_df)]
    test_df = combined_df.iloc[len(train_df):]

    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    # Min-Max Scaling (Crucial for Neural Networks)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Step 3: Training Random Forest (Shallow Learning) ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    start_time = time.time()
    rf_model.fit(X_train_scaled, y_train)
    rf_train_time = time.time() - start_time

    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    print(f"Random Forest Training Time: {rf_train_time:.2f} seconds")
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_pred, target_names=['Normal', 'Attack']))


    print("\n--- Step 4: Training 1D Convolutional Neural Network (Deep Learning) ---")
    # Reshape data for 1D CNN: (samples, time steps, features)
    X_train_cnn = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_cnn = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(64, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(1, activation='sigmoid')) # Binary classification

    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    # Training the CNN (10 Epochs)
    history = cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=256, validation_split=0.2, verbose=1)
    cnn_train_time = time.time() - start_time

    # Predict and evaluate
    cnn_pred_probs = cnn_model.predict(X_test_cnn)
    cnn_pred = (cnn_pred_probs > 0.5).astype(int).flatten()
    cnn_accuracy = accuracy_score(y_test, cnn_pred)

    print(f"\nCNN Training Time: {cnn_train_time:.2f} seconds")
    print(f"CNN Accuracy: {cnn_accuracy * 100:.2f}%")
    print("CNN Classification Report:")
    print(classification_report(y_test, cnn_pred, target_names=['Normal', 'Attack']))

    print("\n--- Step 5: Generating Visualizations ---")
    # 1. Plot Training vs Validation Accuracy for CNN
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.savefig('cnn_accuracy_curve.png') # Saves the image automatically
    print("Saved 'cnn_accuracy_curve.png'")
    plt.show()

    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rf_cm = confusion_matrix(y_test, rf_pred)
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    axes[0].set_title('Random Forest Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    cnn_cm = confusion_matrix(y_test, cnn_pred)
    sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1], xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    axes[1].set_title('CNN Confusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png') # Saves the image automatically
    print("Saved 'confusion_matrices.png'")
    plt.show()

if __name__ == "__main__":
    main()
