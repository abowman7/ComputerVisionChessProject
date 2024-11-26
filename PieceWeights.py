import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def load_images_from_folders(base_folder):
    # Initialize lists to hold images and their labels
    images = []
    labels = []
    
    # Loop through each folder in the base folder
    for root, dirs, files in os.walk(base_folder):
        # Skip the base folder itself, only process subfolders
        if root == base_folder:
            continue
        
        # Extract the label from the folder name (this will be the folder name itself)
        folder_label = os.path.basename(root)
        
        # Get all image files (you can modify the extension as needed)
        image_files = glob.glob(os.path.join(root, '*.png')) + glob.glob(os.path.join(root, '*.jpg')) + glob.glob(os.path.join(root, '*.jpeg'))

        # Loop through each image file and load it
        for image_file in image_files:
            try:
                # Open the image using Pillow
                img = Image.open(image_file)
                # Convert image to numpy array
                img_array = np.array(img)
                
                # Append the image and its label
                images.append(img_array)
                labels.append(folder_label)
            except Exception as e:
                print(f"Could not process image {image_file}: {e}")
    
    return images, labels

# Define the base folder path
base_folder = 'training_tiles'

# Call the function
images, labels = load_images_from_folders(base_folder)

# Now `images` holds all the image arrays, and `labels` holds corresponding folder labels
print(f"Loaded {len(images)} images with {len(set(labels))} unique labels.")
# for label in labels:
#     print(label)
# for image in images:
#     plt.imshow(image)
#     plt.show()

# Normalize pixel values to range [0, 1]
images = np.array(images).astype('float32') / 255.0

# Step 3: Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded)

# Step 4: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)

# Step 5: Build a simple CNN model
model = Sequential()

# Add more convolutional layers to capture more complex features
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(len(np.unique(labels)), activation='softmax'))

# Step 6: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

predictions = model.predict(X_test)  # Get the predicted probabilities
for i in range(len(predictions)):
    print("=============")
    print("Real Label: ", y_test[i], " - ", np.argmax(y_test[i]))
    print("Predicted: ", predictions[i], " - ", np.argmax(predictions[i]))
    plt.imshow(X_test[i])
    plt.show()

# Step 8: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test accuracy: {accuracy * 100:.2f}%")

# Optionally, plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
#plt.show()

# # Load the pre-trained VGG16 model (without the top classification layers)
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 1))

# # Freeze the base model layers so they are not trained
# base_model.trainable = False

# # Add custom layers for chess piece classification
# x = Flatten()(base_model.output)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(len(np.unique(labels)), activation='softmax')(x)  # Number of classes equals the number of unique chess pieces

# # Define the model
# model = Model(inputs=base_model.input, outputs=x)

# # Compile the model
# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))