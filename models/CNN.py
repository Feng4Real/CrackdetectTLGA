#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.optimizers import Adam
import pickle

# Data preparation 
train_data_path = "../crack_images"
test_data_path = "../crack_test"
train_datagen = ImageDataGenerator(validation_split=0.3, preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(train_data_path, target_size=(224, 224), batch_size=64, shuffle=True, class_mode='categorical', subset='training')
validation_generator = train_datagen.flow_from_directory(train_data_path, target_size=(224, 224), batch_size=64, class_mode='categorical', subset='validation')
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_data_path,  
    target_size=(224, 224),
    batch_size=64,
    shuffle=False,
    class_mode='categorical'
)


# CNN structure
model = Sequential()
model.add(Conv2D(64,3,padding="same", activation="relu", input_shape = (224, 224, 3)))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(128, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(2, activation="softmax"))

model.summary()
opt = Adam(learning_rate=1e-5)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) 
history = model.fit(
        train_generator,
        epochs=30,  
        validation_data=validation_generator,
        verbose=1
    )


with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
model.save('pureCNNmodel.h5')



predictions = model.predict(test_generator)
predicted_classes = predictions.argmax(axis=1)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())  # List of class labels

loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")



cm = confusion_matrix(true_classes, predicted_classes)
print(cm)
print(classification_report(true_classes, predicted_classes, target_names=class_labels))


