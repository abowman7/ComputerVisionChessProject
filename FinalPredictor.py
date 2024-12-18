import tensorflow as tf
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import io
from GradientFuncs import gradientx, gradienty
from IPython.display import clear_output, Image, display
import scipy.ndimage as nd
import scipy.signal
from movePredictor import generateMove
from PieceWeights import load_images_from_folders, cnn, train_model
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
np.set_printoptions(suppress=True)

# input the image array and output the image
def showImage(a, fmt='jpeg', rng=[0,1]):
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  PIL.Image.fromarray(a).show()

#input file Here:
#img_file = 'Game7.jpeg'
img_file = input('Path to input image: ')
turn = input('Which players turn? (w/b): ')
img = PIL.Image.open(f"input_boards/{img_file}")
#img = PIL.Image.open("randfens/randfenA.png")


#convert from png to jpg if png
if img.mode == 'RGBA':
    img = img.convert('RGB')

# turn to grayscale image
a = np.asarray(img.convert("L"), dtype=np.float32)

#gradient x to help get vertical grid for chess board
grad_x = gradientx(a)
#gradient y to help get horizontal grid for chess board
grad_y = gradienty(a)

#show gradients if wanted
#showImage(grad_x.numpy(), rng=[-255,255])
#showImage(grad_y.numpy(), rng=[-255,255])

#separate gradients between positive and negative
#vertical gradient
gxPos = tf.clip_by_value(grad_x, 0., 255., name="dx_positive")
gxNeg = tf.clip_by_value(grad_x, -255., 0., name='dx_negative')
#horizontal gradient
gyPos = tf.clip_by_value(grad_y, 0., 255., name="dy_positive")
gyNeg = tf.clip_by_value(grad_y, -255., 0., name='dy_negative')
#get hough gradients (+ * -) / axis^2
houghgx = tf.reduce_sum(gxPos, 0) * tf.reduce_sum(-gxNeg, 0) / (a.shape[0]*a.shape[0])
houghgy = tf.reduce_sum(gyPos, 1) * tf.reduce_sum(-gyNeg, 1) / (a.shape[1]*a.shape[1])

# make threshhold half of maximum hough gradient value for each
houghgx_thr = tf.reduce_max(houghgx) / 2
houghgy_thr = tf.reduce_max(houghgy) / 2

def pruneLines(lineset):
    linediff = np.diff(lineset)
    x = 0
    cnt = 0
    start_pos = 0
    for i, line in enumerate(linediff):
        # Within 5 px of the other (allowing for minor image errors)
        if np.abs(line - x) < 5:
            cnt += 1
            if cnt == 5:
                end_pos = i+2
                return lineset[start_pos:end_pos]
        else:
            cnt = 0
            x = line
            print(i, x)
            start_pos = i
    return lineset

def skele1d(arr):
    _arr = arr.copy() # copy array
    # go forwards, shift right if values are the same
    for i in range(_arr.size-1):
        if arr[i] <= _arr[i+1]:
            _arr[i] = 0
    
    # go backwards, shift left
    for i in np.arange(_arr.size-1, 0,-1):
        if _arr[i-1] > _arr[i]:
            _arr[i] = 0
    return _arr

def gridLines(hdx, hdy, hdx_thr, hdy_thr):
    """Returns pixel indices for the 7 internal chess lines in x and y axes"""
    # gaussian to add blur
    gauss = scipy.signal.windows.gaussian(21,4., sym = True)
    #normalize gaussian
    gauss /= np.sum(gauss)

    # blur vertical and horizontal
    bx = np.convolve(hdx > hdx_thr, gauss, mode='same')
    by = np.convolve(hdy > hdy_thr, gauss, mode='same')

    skel_x = skele1d(bx)
    skel_y = skele1d(by)

    # find grid lines from 1d skeleton arrays
    vertLines = np.where(skel_x)[0] # vertical lines
    horLines = np.where(skel_y)[0] # horizontal lines
    
    # prune lines too close to eachother
    vertLines = pruneLines(vertLines)
    horLines = pruneLines(horLines)
    
    #is_match = len(vertLines) == 7 and len(horLines) == 7 and checkMatch(vertLines) and checkMatch(horLines)
    
    return vertLines, horLines

# Get chess lines
vertLines, horLines = gridLines(houghgx.numpy().flatten(), \
                                           houghgy.numpy().flatten(), \
                                           houghgx_thr.numpy()*.9, \
                                           houghgy_thr.numpy()*.9)


#split chess board into all 64 chess tiles and return array of each tile
def sliceTiles(a, vertLines, horLines):
    # get average tile size
    xsize = np.int32(np.round(np.mean(np.diff(vertLines))))
    ysize = np.int32(np.round(np.mean(np.diff(horLines))))
    
    # pad edges for chess board images that aren't properly cropped
    rightXPad = 0
    leftXPad = 0
    rightYPad = 0
    leftYPad = 0
    #get image pads
    if vertLines[0] - xsize < 0:
        leftXPad = np.abs(vertLines[0] - xsize)
    if vertLines[-1] + xsize > a.shape[1]-1:
        rightXPad = np.abs(vertLines[-1] + xsize - a.shape[1])
    if horLines[0] - ysize < 0:
        leftYPad = np.abs(horLines[0] - ysize)
    if horLines[-1] + xsize > a.shape[0]-1:
        rightYPad = np.abs(horLines[-1] + ysize - a.shape[0])
    
    # pad image
    a2 = np.pad(a, ((leftYPad,rightYPad), (leftXPad,rightXPad)), mode='edge')
    
    #get the beginning and end of each tile along x and y axis
    setsx = np.hstack([vertLines[0]-xsize, vertLines, vertLines[-1]+xsize]) + leftXPad
    setsy = np.hstack([horLines[0]-ysize, horLines, horLines[-1]+ysize]) + leftYPad

    a2 = a2[setsy[0]:setsy[-1], setsx[0]:setsx[-1]]
    setsx -= setsx[0]
    setsy -= setsy[0]
    #     showImage(a2, rng=[0,255])    
    #     print "X:",setsx
    #     print "Y:",setsy
    
    # tiles holds each chessboard square
    #     print "Square size: [%g, %g]" % (ysize, xsize)
    tiles = np.zeros([np.round(ysize), np.round(xsize), 64,1],dtype=np.uint8)
    
    # each row 1-8
    for i in range(0,8):
        # each column a-h
        for j in range(0,8):
            rightXPad = 0
            leftXPad = 0
            rightYPad = 0
            leftYPad = 0

            # vertical line bounds
            x1 = setsx[i]
            x2 = setsx[i+1]

            if (x2-x1) > xsize:
                if i == 7:
                    x1 = x2 - xsize
                else:
                    x2 = x1 + xsize
            elif (x2-x1) < xsize:
                if i == 7:
                    # assign pad
                    rightXPad = xsize-(x2-x1)
                else:
                    # assign pad
                    leftXPad = xsize-(x2-x1)
            # horizontal line bounds
            y1 = setsy[j]
            y2 = setsy[j+1]

            if (y2-y1) > ysize:
                if j == 7:
                    y1 = y2 - ysize
                else:
                    y2 = y1 + ysize
            elif (y2-y1) < ysize:
                if j == 7:
                    # assign pad
                    rightYPad = ysize-(y2-y1)
                else:
                    # assign pad
                    leftYPad = ysize-(y2-y1)
            # slicing a, rows sliced with horizontal lines, cols by vertical lines so reversed
            # change order so its A1,B1...H8 for a white-aligned board
            tiles[:,:,(7-j)*8+i,0] = np.pad(a2[y1:y2, x1:x2],((leftYPad,rightYPad),(leftXPad,rightXPad)), mode='edge')
    return tiles


tiles = sliceTiles(a, vertLines, horLines)

ttiles = np.array([np.asarray(PIL.Image.fromarray(tiles[:,:,0,0]).resize([32,32], PIL.Image.ADAPTIVE))])
for i in range(1,64):
    ttiles = np.append(ttiles, [np.asarray(PIL.Image.fromarray(tiles[:,:,i,0]).resize([32,32], PIL.Image.ADAPTIVE))], axis=0)
    
tiles = ttiles


# Define the base folder path
base_folder = 'training_tiles'
# ima, labes = load_images_from_folders(base_folder)
model = load_model("best_model.keras")
#test_predictions = train_model(model, ima, labes)

predictions = model.predict(tiles)

board_array = np.zeros((8,8))

for i in range(8):
    for j in range(8):
        board_array[7-i][j] = np.argmax(predictions[i*8+j])

print(board_array)

#tiles index goes A1, B1, C1, D1, E1, F1, H1, A2, B2, ...
# Index to piece label Cheat Sheet:
# 0 - B Bishop, 1 - B King, 2 - B Knight, 3 - B Pawn, 4 - B Queen, 5 - B Rook, 6 = Blank Tile
# 7 - W Bishop, 8 - W King, 9 - W Knight, 10 - W Pawn, 11 - W Queen, 12 - W Rook

#this is an example array for now, replace with generated array later
#boardArray = [[8,9,10,11,12,10,9,8],[7,7,7,7,7,7,7,7],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1],[2,3,4,5,6,4,3,2]]
#turn = 'w' #allow this to be selected as a user controlled input variable
move = generateMove(board_array, turn)
print(move)