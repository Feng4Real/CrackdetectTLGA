#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.optimizers import Adam
import pickle
from keras.applications import ResNet50
from keras.models import Model
from sklearn.metrics import classification_report,confusion_matrix

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
    shuffle=False,  # No shuffling for the test set
    class_mode='categorical'
)



base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the base layers
base_model.trainable = False
x = base_model.output
x = Flatten()(x)
x = Dense(2, activation='softmax')(x)

# Define the model with the base model and custom output layer
model = Model(inputs=base_model.input, outputs=x)

# Model compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
opt = Adam(learning_rate=1e-5)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) 
model.summary()
history = model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        verbose=1
    )

# save history and model object
with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
model.save('pureTLmodel.h5')


# Make predictions on the test data
predictions = model.predict(test_generator)
predicted_classes = predictions.argmax(axis=1)

# Get the true labels
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())  # List of class labels

# Evaluate model on test data
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


cm = confusion_matrix(true_classes, predicted_classes)
print(cm)
print(classification_report(true_classes, predicted_classes, target_names=class_labels))


