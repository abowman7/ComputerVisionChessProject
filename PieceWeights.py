import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
#from tensorflow import Sequential, save_model
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
    "BlackBishops", #1
    "BlackKings", #2
    "BlackKnights", #3
    "BlackPawns", #4
    "BlackQueens", #5
    "BlackRooks", #6
    "BlankTiles", #7
    "WhiteBishops", #8
    "WhiteKings", #9
    "WhiteKnights", #10
    "WhitePawns", #11
    "WhiteQueen", #12
    "WhiteRooks" #13
    ]
num_classes = len(classifications)

classes = {
    "BlackBishops": 10,
    "BlackKings": 12,
    "BlackKnights": 9,
    "BlackPawns": 7,
    "BlackQueens": 11,
    "BlackRooks": 8,
    "BlankTiles": 0,
    "WhiteBishops": 4,
    "WhiteKings": 6,
    "WhiteKnights": 3,
    "WhitePawns": 1,
    "WhiteQueens": 5,
    "WhiteRooks": 2
}


def load_images_from_folders(path):
    images = []
    labels = []
    
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

def cnn():
    # Step 5: Build a simple CNN model
    model = Sequential()

    # Add more convolutional layers to capture more complex features
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model

def train_model(model, images, labels):

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
   
    save_best_model = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    
    # Step 7: Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[save_best_model])

    predictions = model.predict(X_test)  # Get the predicted probabilities
    # for i in range(len(predictions)):
    #     print("=============")
    #     print("Real Label: ", y_test[i], " - ", np.argmax(y_test[i]))
    #     print("Predicted: ", predictions[i], " - ", np.argmax(predictions[i]))
    #     plt.imshow(X_test[i])
    #     plt.show()

    # Step 8: Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, callbacks=[save_best_model])

    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Optionally, plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    return predictions
    #plt.show()

def cnn_testing():
    images, labels = load_images_from_folders("training_tiles/")
    model = cnn()
    preds = train_model(model, images, labels)
cnn_testing()