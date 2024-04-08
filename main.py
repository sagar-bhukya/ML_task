# Required libraries
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from fastapi import FastAPI
from pydantic import BaseModel

# Load dataset
with open("dataset.json", "r") as f:
    dataset = json.load(f)

# Data preprocessing
external_statuses = [d["externalStatus"] for d in dataset]
internal_statuses = [d["internalStatus"] for d in dataset]

# Convert external statuses to numerical values
external_status_mapping = {status: i for i, status in enumerate(set(external_statuses))}
encoded_external_statuses = [external_status_mapping[status] for status in external_statuses]

# Convert internal statuses to numerical values
internal_status_mapping = {status: i for i, status in enumerate(set(internal_statuses))}
encoded_internal_statuses = [internal_status_mapping[status] for status in internal_statuses]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_external_statuses, encoded_internal_statuses, test_size=0.2, random_state=42)

# Model Development
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(len(set(internal_statuses)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(np.array(X_test), np.array(y_test))
print("Accuracy:", accuracy)

# API Development
app = FastAPI()

class Item(BaseModel):
    externalStatus: str

@app.post("/predict/")
def predict(item: Item):
    external_status = item.externalStatus
    encoded_status = external_status_mapping.get(external_status, -1)
    if encoded_status == -1:
        return {"error": "Invalid external status"}

    prediction = model.predict(np.array([encoded_status]))
    predicted_internal_status = list(internal_status_mapping.keys())[np.argmax(prediction)]
    return {"externalStatus": external_status, "predictedInternalStatus": predicted_internal_status}

