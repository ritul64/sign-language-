import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forces TensorFlow to use only the CPU

# Paths to dataset
train_dir = r"C:\Users\ritul\OneDrive\Desktop\projects\sign language\dataset\asl_alphabet_train"  //YOUR OWN PATH
test_dir = r"C:\Users\ritul\OneDrive\Desktop\projects\sign language\dataset\asl_alphabet_test"    

# Image properties
IMG_SIZE = 128  # Resize all images to 128x128
BATCH_SIZE = 32  # Batch size

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=(0.8, 1.2)
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Compute class weights for imbalanced data
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Model architecture
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model with an SGD optimizer and gradient clipping
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for better training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(r"C:\Users\ritul\OneDrive\Desktop\projects\sign language\Model\keras_model.h5",
                             monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)

# Optional: Add a learning rate scheduler
lr_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (-epoch / 10))

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoint, reduce_lr, lr_scheduler]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save(r"C:\Users\ritul\OneDrive\Desktop\projects\sign language\Model\keras_model.h5") //saved file 

