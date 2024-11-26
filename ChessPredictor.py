import tensorflow as tf
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import io
from GradientFuncs import gradientx, gradienty
from IPython.display import clear_output, Image, display
import scipy.ndimage as nd
import scipy.signal
np.set_printoptions(suppress=True)

# input the image array and output the image
def showImage(a, fmt='jpeg', rng=[0,1]):
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  PIL.Image.fromarray(a).show()

img_file = 'BoardSetup.jpg'
img = PIL.Image.open(f"input_boards/{img_file}")

#convert from png to jpg if png
if img.mode == 'RGBA':
    img = img.convert('RGB')
#display the image
showImage(np.asarray(img), rng=[0,255])

# turn to grayscale image
a = np.asarray(img.convert("L"), dtype=np.float32)

# show grayscale image 
showImage(a, rng=[0,255])

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

#plot hough gradients
fig, (p1, p2) = plt.subplots(1,2,sharey=True, figsize=(15,5))
#verticals plot
p1.plot(houghgx.numpy())
p1.axhline(houghgx_thr.numpy(), lw=2,linestyle=':',color='r')
p1.set_xlabel('pixel')
p1.set_title('hough x')
p1.set_xlim(0,a.shape[1])
#horizontals plot
p2.plot(houghgy.numpy())
p2.axhline(houghgy_thr.numpy(), lw=2,linestyle=':',color='r')
p2.set_xlim(0,a.shape[0])
p2.set_title('hough y')
p2.set_xlabel('pixel')
#show plots
plt.show()

# def checkMatch(lineset):
#     """Checks whether there exists 7 lines of consistent increasing order in set of lines"""
#     linediff = np.diff(lineset)
#     x = 0
#     cnt = 0
#     for line in linediff:
#         # Within 5 px of the other (allowing for minor image errors)
#         if np.abs(line - x) < 5:
#             cnt += 1
#         else:
#             cnt = 0
#             x = line
#     return cnt == 5

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

print("X",vertLines, np.diff(vertLines))
print("Y",horLines, np.diff(horLines))
if vertLines.shape[0] == 7 and horLines.shape[0] == 7:
    print("Chessboard found")
else:
    print("Couldn't find Chessboard")

# plot blurred hough gradients with lines
fig, (p1, p2) = plt.subplots(1,2, figsize=(20,5))
#vertical lines
p1.plot(houghgx.numpy())
p1.axhline(houghgx_thr.numpy(), lw=2,linestyle=':',color='r')
p1.set_xlabel('pixel')
p1.set_title('hough x')
p1.set_xlim(0,a.shape[1])
#horizontal lines
p2.plot(houghgy.numpy())
p2.axhline(houghgy_thr.numpy(), lw=2,linestyle=':',color='r')
p2.set_xlim(0,a.shape[0])
p2.set_xlabel('pixel')
p2.set_title('hough y')

# plot lines in graph
if len(vertLines < 20):
    for hx in vertLines: 
        p1.axvline(hx,color='orange')
if len(horLines < 20):
    for hy in horLines:     
        p2.axvline(hy,color='orange')

plt.show()
plt.imshow(img)
for hx in vertLines:    #vertical grid lines
    plt.axvline(hx, color='r', lw=2)

for hy in horLines:     #horizontal grid lines
    plt.axhline(hy, color='lime', lw=2)

plt.show()
#display each line's pixel value
#print("Vertical Lines: ", vertLines, np.diff(vertLines))
#print("Horizontal: ", horLines, np.diff(horLines))

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
    tiles = np.zeros([np.round(ysize), np.round(xsize), 64],dtype=np.uint8)
    
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
            tiles[:,:,(7-j)*8+i] = np.pad(a2[y1:y2, x1:x2],((leftYPad,rightYPad),(leftXPad,rightXPad)), mode='edge')
    return tiles


tiles = sliceTiles(a, vertLines, horLines)
# if is_match:
#     # Possibly check np.std(np.diff(vertLines)) for variance etc. as well/instead
#     print("7 horizontal and vertical lines found, slicing up squares")
#     squares = sliceTiles(a, vertLines, horLines)
#     print("Tiles generated: (%dx%d)*%d" % (squares.shape[0], squares.shape[1], squares.shape[2]))
# else:
#     print("Number of lines not equal to 7")

letters = 'ABCDEFGH'

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
img_save_dir = dir_path+("/output_tiles/squares_%s" % img_file[:-4])

if False:
    print("No squares to save")
else:
    #create directory if needed
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
        print("Created dir %s" % img_save_dir)
    #iterate through tiles and save them
    for i in range(64):
        #save tile file
        sqr_filename = "%s/%s_%s%d.jpg" % (img_save_dir, img_file[:-4], letters[i%8], i/8+1)
        if i % 8 == 0:
            print("#%d: saving %s..." % (i, sqr_filename))
        
        # resize image to 32x32 and save
        PIL.Image.fromarray(tiles[:,:,i]) \
            .resize([32,32], PIL.Image.ADAPTIVE) \
            .save(sqr_filename)