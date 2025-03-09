import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # สำหรับการบันทึกโมเดล
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = pd.read_csv('C:/Users/kittisak/OneDrive/Documents/AI/web/train.csv')

# Data Preparation: Split the data into features (X) and target (y)
X = dataset.drop('price_range', axis=1)
y = dataset['price_range']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

# Normalize the data (Optional but recommended for Neural Networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Train other models ---

# Training & Evaluation with Linear Regression Model
lm = LinearRegression()
lm.fit(X_train, y_train)
lm_score = lm.score(X_test, y_test)

# Training & Evaluation with KNN Model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)

# Elbow method for KNN to find optimal K value (optional step)
error_rate = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Train & Evaluate Logistic Regression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
logmodel_score = logmodel.score(X_test, y_test)

# Train & Evaluate Decision Tree Model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dtree_score = dtree.score(X_test, y_test)

# Train & Evaluate Random Forest Model
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfc_score = rfc.score(X_test, y_test)

# --- Train Neural Network Model ---
# Build Neural Network Model
model = Sequential()

# Input layer (first layer) and hidden layers
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))  # Dropout to avoid overfitting

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(1, activation='linear'))  # Linear activation for regression output

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on test data
test_loss = model.evaluate(X_test, y_test)
print(f"Neural Network Test Loss: {test_loss}")

# Save the trained Neural Network model to a file
model.save('mobile_price_model.keras')

# --- Print scores and results ---
# Result Summary
print(f"Linear Regression Score: {lm_score}")
print(f"KNN Score: {knn_score}")
print(f"Logistic Regression Score: {logmodel_score}")
print(f"Decision Tree Score: {dtree_score}")
print(f"Random Forest Score: {rfc_score}")

# Best Model: KNN & Linear Regression performed the best
# KNN results (optional step)
knn_pred = knn.predict(X_test)
print(classification_report(y_test, knn_pred))
cm = confusion_matrix(y_test, knn_pred)
print(cm)

# Price prediction of Test.csv Using KNN for Prediction
data_test = pd.read_csv('C:/Users/kittisak/OneDrive/Documents/AI/web/test.csv')
data_test = data_test.drop('id', axis=1)  # Drop the 'id' column
predicted_price = knn.predict(data_test)

# Adding Predicted Price to Test.csv
data_test['price_range'] = predicted_price
data_test.to_csv('predicted_test_results.csv', index=False)  # Save to CSV if needed

# Save the KNN model to a file for future use
joblib.dump(knn, 'knn_model.pkl')

# Optionally, load the model back for predictions in the future:
# knn_loaded = joblib.load('knn_model.pkl')
