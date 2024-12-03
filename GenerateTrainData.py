import numpy as np
import PIL
import os

def getRandomFEN():
    fen_chars = list('1KQRBNPkqrbnp')
    pieces = np.random.choice(fen_chars, 64)
    fen = '/'.join([''.join(pieces[i*8:(i+1)*8]) for i in range(8)])
    # can append ' w' or ' b' for white/black to play, defaults to white
    return fen

import urllib, cStringIO
def generateRandomBoards(n, outfolder, img_url_template, fen_chars='1KQRBNPkqrbnp'):
    """Given chess diagram template url, generate n random FEN diagrams from url and save images to outfolder"""
    # http://www.jinchess.com/chessboard/?p=rnbqkbnrpppppppp----------P----------------R----PP-PPPPPRNBQKBNR
    # http://www.apronus.com/chess/stilldiagram.php?d=DRNBQKBNRPP_PPPPP__P______P___________p_____k____pppQp_pprnbq_bnr0
    # No / separators for either choice
    
    # Create output folder as needed
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    
    for i in range(n):
        fen_chars = list(fen_chars)
        fen_arr = np.random.choice(fen_chars, 64)
        fen = ''.join(fen_arr)
        img_url = img_url_template % fen
        img = PIL.Image.open(cStringIO.StringIO(urllib.urlopen(img_url).read()))
        if 'apronus' in img_url_template:
            # need to flip FEN file order since the are 1-8 vs 8-1 of normal FEN.
            fen_arr = np.hstack(np.split(fen_arr,8)[::-1])
        
        # Replace - or _ with 1 to be consistent with actual FEN notation
        fen_arr[fen_arr == fen_chars[0]] = '1'
        
        # Add - between sets of 8 to be consistent with saved file format (later converted to / again for analysis link)
        fen = '-'.join(map(''.join, np.split(fen_arr, 8)))
        
        img.save(os.path.join(outfolder, fen+'.png'))
#
generateRandomBoards(20,'chessboards/train_images', "http://www.jinchess.com/chessboard/?p=%s", '-KQRBNPkqrbnp')
generateRandomBoards(20,'chessboards/train_images', "http://www.apronus.com/chess/stilldiagram.php?d=_%s", '_KQRBNPkqrbnp')