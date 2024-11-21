import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import zipfile
import warnings
import pathlib
warnings.filterwarnings('ignore')

# Run on google colab or locally or on Kaggle 

if os.path.exists('/kaggle/input'):
    print('Running on Kaggle...')
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            # Read the CSV file from Kaggle
            df = pd.read_csv(os.path.join(dirname, filename))
    print("Data loaded successfully from Kaggle!")
else:
    try:
        from google.colab import drive
        print('Running on CoLab...')
        drive.mount('/content/drive')
        # Read the csv file from Collab
        df = pd.read_csv('/content/drive/My Drive/ClassicHit.csv')
        print("Done!")
    except:
        print("Not running on Kaggle")
        # Download the data
        # This is the link for the data set
        # https://www.kaggle.com/datasets/thebumpkin/10400-classic-hits-10-genres-1923-to-2023/data
        !kaggle datasets download -d thebumpkin/10400-classic-hits-10-genres-1923-to-2023
    
        # Jupyter Notebook's path
        desktop = pathlib.Path.home() / 'Desktop'
        os.chdir(desktop) # This automatically transfer you to Desktop (suppose that your jupyter notebook is on Desktop)
    
        # Unzip the files
        with zipfile.ZipFile('./10400-classic-hits-10-genres-1923-to-2023.zip', 'r') as zip_ref:
            zip_ref.extractall('./Genre_Dataset')
    
    
        print('Changing Directory!')
        os.chdir('./Genre_Dataset')
        print('Current Directory: ', os.getcwd())
    
        # Read the csv from Desktop locally
        df = pd.read_csv('ClassicHit.csv')
    
        print("Done!")

# Display basic info about the dataset
print(df.info())
print(f"Number of unique genres: {df['Genre'].nunique()}")

# Preprocessing
# Verify the columns in the DataFrame
print("Columns in the dataset:", df.columns)

# Convert duration to seconds if the column exists
if 'Duration' in df.columns:
    df['Duration'] = df['Duration'] / 1000

# Drop unnecessary columns if they exist
df = df.drop(['Track', 'Artist', 'Year'], axis=1, errors='ignore')

# Verify the changes
print("Updated columns:", df.columns)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Genre'] = label_encoder.fit_transform(df['Genre'])
print(f"Encoded genres: {list(label_encoder.classes_)}")

# Define features and target
X = df.drop('Genre', axis=1)
y = df['Genre']

# Scale features
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Identify non-numeric columns
print("Data types of columns:")
print(df.dtypes)

# Drop or encode non-numeric columns as needed
if 'Key' in df.columns or 'Mode' in df.columns or 'Time_Signature' in df.columns:
    df['Key'] = LabelEncoder().fit_transform(df['Key'])
    df['Mode'] = LabelEncoder().fit_transform(df['Mode'])
    df['Time_Signature'] = LabelEncoder().fit_transform(df['Time_Signature'])

# Ensure all columns are numeric
X_numeric = df.select_dtypes(include=[np.number])
print("Columns selected for scaling:", X_numeric.columns)

# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Verify the scaled data
print("Scaled data shape:", X_scaled.shape)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Utility function for evaluating models
def evaluate_model(name, y_test, y_pred):
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ================================
# 1. KNN Classifier
# ================================
knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid=knn_params, cv=5, scoring='accuracy')
knn_grid.fit(X_train, y_train)
knn_best_model = knn_grid.best_estimator_

y_pred_knn = knn_best_model.predict(X_test)
evaluate_model("KNN", y_test, y_pred_knn)

# ================================
# 2. Random Forest Classifier
# ================================
rf_params = {'n_estimators': [100, 300, 500], 'max_depth': [4, 6, 8], 'criterion': ['gini', 'entropy']}
rf_grid = GridSearchCV(RandomForestClassifier(), param_grid=rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)
rf_best_model = rf_grid.best_estimator_

y_pred_rf = rf_best_model.predict(X_test)
evaluate_model("Random Forest", y_test, y_pred_rf)

# ================================
# 3. XGBoost Classifier
# ================================
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
evaluate_model("XGBoost", y_test, y_pred_xgb)

# ================================
# 4. Convolutional Neural Network (CNN)
# ================================

# Ensure proper reshaping for CNN input
# Determine grid dimensions based on the number of features (16 in this case)
num_features = X_train.shape[1]
height, width = 4, 4  # Adjust dimensions to fit 16 features (4x4 grid)

# Validate that reshaping dimensions are correct
if height * width != num_features:
    raise ValueError(f"Cannot reshape data of {num_features} features into {height}x{width} grid")

# Reshape features for CNN (4x4 grid with 1 channel)
X_train_cnn = X_train.reshape(-1, height, width, 1)
X_test_cnn = X_test.reshape(-1, height, width, 1)

# Define CNN architecture
cnn_model = Sequential([
    Conv2D(32, (2, 2), activation='relu', input_shape=(height, width, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile CNN model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train CNN model
cnn_model.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test), epochs=20, batch_size=32)

# Evaluate CNN model
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test)
print(f"CNN Accuracy: {cnn_accuracy:.3f}")

# ================================
# 4. Recurrent Neural Network (RNN)
# ================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Ensure proper reshaping for RNN input
# Assume we have 16 features and reshape them into a sequence of 4 timesteps with 4 features per timestep
num_features = X_train.shape[1]
timesteps, features_per_timestep = 4, 4  # Adjust dimensions to fit the dataset

# Validate that reshaping dimensions are correct
if timesteps * features_per_timestep != num_features:
    raise ValueError(f"Cannot reshape data of {num_features} features into {timesteps}x{features_per_timestep} sequence")

# Reshape features for RNN (4 timesteps, 4 features per timestep)
X_train_rnn = X_train.reshape(-1, timesteps, features_per_timestep)
X_test_rnn = X_test.reshape(-1, timesteps, features_per_timestep)

# Normalize the input data
X_train_rnn = X_train_rnn / 255.0
X_test_rnn = X_test_rnn / 255.0

# Compute class weights
unique_classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
class_weights = dict(zip(unique_classes, class_weights))

# Define RNN architecture
rnn_model = Sequential([
    # First RNN layer
    SimpleRNN(64, activation='relu', input_shape=(timesteps, features_per_timestep), return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    # Second RNN layer
    SimpleRNN(64, activation='relu', return_sequences=False),
    BatchNormalization(),
    Dropout(0.4),

    # Fully connected layer
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    # Output layer
    Dense(len(unique_classes), activation='softmax')
])

# Compile RNN model
rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train RNN model
rnn_model.fit(X_train_rnn, y_train, validation_data=(X_test_rnn, y_test),
              epochs=50, batch_size=32, class_weight=class_weights, callbacks=[lr_reduction, early_stopping])

# Evaluate RNN model
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test_rnn, y_test)
print(f"RNN Accuracy: {rnn_accuracy:.3f}")

# ================================
# 5. Model Comparison
# ================================
models = ["KNN", "Random Forest", "XGBoost", "CNN"]
accuracies = [
    accuracy_score(y_test, y_pred_knn),
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_xgb),
    cnn_accuracy
]

plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.show()
