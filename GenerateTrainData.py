import urllib.request
import requests
import platform
import numpy as np
import os
import PIL
from PIL import Image
from IPython.display import clear_output, Image, display
import io
import urllib
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import matplotlib.pyplot as plt
from GradientFuncs import gradientx, gradienty
import tensorflow as tf
import scipy.signal

def getRandomFEN():
    fen_chars = list('1KQRBNPkqrbnp')
    pieces = np.random.choice(fen_chars, 64)
    fen = '/'.join([''.join(pieces[i*8:(i+1)*8]) for i in range(8)])
    # can append ' w' or ' b' for white/black to play, defaults to white
    return fen

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
    
    # tiles holds each chessboard square
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
            # change order so its A1,B1...G8, H8 for a well-aligned board
            tiles[:,:,(7-j)*8+i] = np.pad(a2[y1:y2, x1:x2],((leftYPad,rightYPad),(leftXPad,rightXPad)), mode='edge')
    return tiles

def main():
    FEN = getRandomFEN()
    print(FEN)
    # Set up URL and output image filename for this run
    url = "http://en.lichess.org/editor/%s" % FEN
    output_filename = "randfens/randfenA.png"

    if not os.path.exists("./randfens/"):  
        os.makedirs("randfens/") 

    if platform.system() == "Windows":
        options = webdriver.EdgeOptions()
        driver = webdriver.Edge(options = options)
    elif platform.system() == "Linux": #I use linux
        options = webdriver.FirefoxOptions()
        driver = webdriver.Firefox(options = options)

    # Open the webpage
    driver.get(url)

    time.sleep(2)

    link_element = driver.find_element(By.XPATH, '//a[text()="SCREENSHOT"]')  # Match the link text
    file_url = link_element.get_attribute('href')

    if file_url:
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(output_filename, 'wb') as file:
                file.write(response.content)
            print('File downloaded successfully!')
        else:
            print(f'Failed to download the file. HTTP status code: {response.status_code}')
    else:
        print('No valid href found.')

    driver.quit()

    img = PIL.Image.open(output_filename)
    #comment this out when you know it works to run quicker

    #plt.imshow(img)
    #plt.show()

    a = np.asarray(img.convert("L"), dtype=np.float32)
    #gradient x to help get vertical grid for chess board
    grad_x = gradientx(a)
    #gradient y to help get horizontal grid for chess board
    grad_y = gradienty(a)

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

    # Get chess lines
    vertLines, horLines = gridLines(houghgx.numpy().flatten(), \
                                            houghgy.numpy().flatten(), \
                                            houghgx_thr.numpy()*.9, \
                                            houghgy_thr.numpy()*.9)

    tiles = sliceTiles(a, vertLines, horLines)

    letters = 'ABCDEFGH'

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_save_dir = dir_path+("/training_tiles/")

    multiFENs = FEN.split('/')
    multiFENs.reverse()
    print(multiFENs)
    count = 0
    for row in multiFENs:
        for piece in row:
            if piece == 'P':
                sqr_filename = "%s/%s/%s.jpg" % (img_save_dir, 'WhitePawns', 'WhitePawns')
            elif piece == 'p':
                sqr_filename = "%s/%s/%s.jpg" % (img_save_dir, 'BlackPawns', 'BlackPawns')
            elif piece == 'R':
                sqr_filename = "%s/%s/%s.jpg" % (img_save_dir, 'WhiteRooks', 'WhiteRooks')
            elif piece == 'r':
                sqr_filename = "%s/%s/%s.jpg" % (img_save_dir, 'BlackRooks', 'BlackRooks')
            elif piece == 'K':
                sqr_filename = "%s/%s/%s.jpg" % (img_save_dir, 'WhiteKings', 'WhiteKings')
            elif piece == 'k':
                sqr_filename = "%s/%s/%s.jpg" % (img_save_dir, 'BlackKings', 'BlackKings')
            elif piece == 'Q':
                sqr_filename = "%s/%s/%s.jpg" % (img_save_dir, 'WhiteQueens', 'WhiteQueens')
            elif piece == 'q':
                sqr_filename = "%s/%s/%s.jpg" % (img_save_dir, 'BlackQueens', 'BlackQueens')
            elif piece == 'B':
                sqr_filename = "%s/%s/%s.jpg" % (img_save_dir, 'WhiteBishops', 'WhiteBishops')
            elif piece == 'b':
                sqr_filename = "%s/%s/%s.jpg" % (img_save_dir, 'BlackBishops', 'BlackBishops')
            elif piece == 'N':
                sqr_filename = "%s/%s/%s.jpg" % (img_save_dir, 'WhiteKnights', 'WhiteKnights')
            elif piece == 'n':
                sqr_filename = "%s/%s/%s.jpg" % (img_save_dir, 'BlackKnights', 'BlackKnights')
            elif piece == '1':
                sqr_filename = "%s/%s/%s.jpg" % (img_save_dir, 'BlankTiles', 'BlankTiles')
            file_counter = 1
            base_filename, file_extension = os.path.splitext(sqr_filename)
            while os.path.exists(sqr_filename):
                sqr_filename = f"{base_filename}{file_counter}{file_extension}"
                file_counter += 1
            PIL.Image.fromarray(tiles[:,:,count]) \
                .resize([32,32], PIL.Image.ADAPTIVE) \
                .save(sqr_filename) 
            count += 1
    
for _ in range(20):
    main()

