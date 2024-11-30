import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import optimizers
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

def load_data():
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

        # Loop through each image file and load it
        for image_file in image_files:
            try:
                # Open the image using Pillow
                img = Image.open(image_file)

                img = image.convert("RGB")
                # Convert image to numpy array
                img_array = np.array(img)
                
                # Append the image and its label
                images.append(img_array)
                labels.append(classes[folder_label])
            except Exception as e:
                print(f"Could not process image {image_file}: {e}")
    
    return images, labels


def cnn():
    model = Sequential()

    model.add(Conv2D(16, (3,3), padding='same', input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3,3), padding='same'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.RMSprop(learning_rate=0.0001, weight_decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    print(model.summary())

    return model

def train_model():
    history_activations = dict()

    model = cnn()
    
    images, labels = load_data()
    x_train = np.array(images)
    y_train = np.array(labels)
    # labels = to_categorical(labels, num_classes)
    # images = np.array(images).astype('float32') / 255.0
    # X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    #y_test = to_categorical(y_test, num_classes)
    # normalize the data
    x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    x_train /= 255
    #x_test /= 255
    save_best_model = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
  

    history_activations[activation] = model.fit(x_train, y_train,
                                              batch_size=32,
                                              epochs=10,
                                              validation_data=(x_train, y_train),
                                              shuffle=True,
                                              callbacks=[save_best_model])

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

train_model()
