import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

# Enable mixed precision (if GPU supports)
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Set paths for your dataset
train_data_dir = 'C:\\Users\\ritul\\OneDrive\\Desktop\\projects\\sign language\\dataset\\asl_alphabet_train\\asl_alphabet_train'

# Ensure the directory exists
if not os.path.exists(train_data_dir):
    print(f"The specified directory does not exist: {train_data_dir}")
    exit(1)

# Parameters
img_height, img_width = 224, 224  # Maintain high resolution
batch_size = 32  # Moderate batch size for accuracy and memory optimization
num_classes = 29  # Number of classes in dataset

# Data Generators with validation split
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% of the data for validation
)

# Training generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',  # 'sparse' is used for integer-encoded labels
    subset='training'  # Set as training data
)

# Validation generator
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',  # 'sparse' is used for integer-encoded labels
    subset='validation'  # Set as validation data
)

# Build the Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze base model for initial training

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax', dtype='float32')  # Specify float32 for the output layer
])

# Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Custom learning rate for stability
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train the Model with Frozen Base
initial_epochs = 15
history = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Unfreeze last layers and fine-tune
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Unfreeze last 20 layers for fine-tuning
    layer.trainable = False

fine_tune_epochs = 5
total_epochs = initial_epochs + fine_tune_epochs
history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr],
    initial_epoch=history.epoch[-1]
)

# Save the Model
model.save('C:\\Users\\ritul\\OneDrive\\Desktop\\projects\\sign language\\Model\\sign_language_model.keras')
