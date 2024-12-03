import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

classifications = [
    "white_pawn_white_space", #1
    "white_pawn_black_space", #2
    "white_rook_white_space", #3
    "white_rook_black_space", #4
    "white_knight_white_space", #5
    "white_knight_black_space", #6
    "white_bishop_white_space", #7
    "white_bishop_black_space", #8
    "white_queen_white_space", #9
    "white_queen_black_space", #10
    "white_king_white_space", #11
    "white_king_black_space", #12
    
    "black_pawn_white_space", #13
    "black_pawn_black_space", #14
    "black_rook_white_space", #15
    "black_rook_black_space", #15
    "black_knight_white_space", #16
    "black_knight_black_space", #17
    "black_bishop_white_space", #18
    "black_bishop_black_space", #20
    "black_queen_white_space", #21
    "black_queen_black_space", #22
    "black_king_white_space", #23
    "black_king_black_space", #24

    "empty_white_space", #25
    "empty_black_space", #26
    ]

num_classes = len(classifications)

classes = {
    "white_pawn_white_space": 0,
    "white_pawn_black_space": 1,
    "white_rook_white_space": 2,
    "white_rook_black_space": 3,
    "white_knight_white_space": 4,
    "white_knight_black_space": 5,
    "white_bishop_white_space": 6,
    "white_bishop_black_space": 7,
    "white_queen_white_space": 8,
    "white_queen_black_space": 9,
    "white_king_white_space": 10,
    "white_king_black_space": 11,
    "black_pawn_white_space": 12,
    "black_pawn_black_space": 13,
    "black_rook_white_space": 14,
    "black_rook_black_space": 15,
    "black_knight_white_space": 16,
    "black_knight_black_space": 17,
    "black_bishop_white_space": 18,
    "black_bishop_black_space": 19,
    "black_queen_white_space": 20,
    "black_queen_black_space": 21,
    "black_king_white_space": 22,
    "black_king_black_space": 23,
    "empty_white_space": 24,
    "empty_black_space": 25,
}

def load_images_from_folders():
    images = []
    labels = []
    
    path = "./training_set"
    for root, dirs, files in os.walk(path):
        # Skip the base folder itself, only process subfolders
        if root == path:
            continue
        
        # Extract the label from the folder name (this will be the folder name itself)
        folder_label = os.path.basename(root)
        
        # Get all image files (you can modify the extension as needed)
        image_files = glob.glob(os.path.join(root, '*.png')) + glob.glob(os.path.join(root, '*.jpg')) + glob.glob(os.path.join(root, '*.jpeg'))
        
        image_files = np.array(image_files)
        np.random.shuffle(image_files)

        # ratio = 0.9
        # split = int(len(image_files) * ratio)

        # Loop through each image file and load it
        for image_file in image_files:
            try:
              # Open the image using Pillow
                img = Image.open(image_file)

                #img = img.convert("gray")
                # Convert image to numpy array
                img_array = np.array(img)
                
                # Append the image and its label
                images.append(img_array)
                labels.append(classes[folder_label])
            except Exception as e:
                print(f"Could not process image {image_file}: {e}")
    
    return images, labels

# Call the function
images, labels = load_images_from_folders()

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
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

opt = keras.optimizers.RMSprop(learning_rate=0.0001, weight_decay=1e-6)

model.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
print(model.summary())

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