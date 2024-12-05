# this code assumes that it is receiving an array matching the shape of a chess board
# in each location should be a number from 0 to 12, where 0 is empty square, 1 or 7 is pawn, 2 or 8 is rook,
# 3 or 9 is knight, 4 or 10 is bishop, 5 or 11 is queen, 6 or 12 is king. 1-6 is White, 7-12 is black
# Location 0,0 in the array is essentiall the top left corner, which is the equivalent to tile a8 on a chess board.
# This code creates a string which matches the structure of Forsyth-Edwards Notation (FEN).
# Thn the FEN string is then passed to stockfish.set_fen_position(FENstring)
# such that after this stockfish.get_best_move() can be run to output a best move in algebraic chess notation
# as a string, which can then be printed/displayed. This chess notation string is the final output of the function.
# e.g. this is the starting FEN position rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
# The code also assumes it is receiving an input of a single character "b" or "w"
# which represents a user input for whose turn it is for the current board position

# To run this code, you must download stockfish from https://stockfishchess.org/download/ and then unzip the .zip file
# then you must provide your own path to stockfish-windows-x86-64-avx2.exe for the code to execute properly
from stockfish import Stockfish

def generateMove(boardArray, turn):
    turn = turn.lower()
    fenString = ""
    openSpaces = 0
    isOpen = False
    # (x,y) = (0,0) represents the top left corner of the array, which is a8
    for x in range(8):
        #x represents numbers 8 through 1 as 0 through 7
        for y in range(8):
            #y represents letters a through h as 0 through 7
            tile = boardArray[x][y]
            if tile == 0: #tile empty
                if isOpen == False:
                    isOpen = True
                    openSpaces = 1
                else:
                    openSpaces += 1
            else: #tile filled
                if isOpen == True:
                    isOpen = False
                    fenString += str(openSpaces)
                    openSpaces = 0
                if tile == 1 or tile == 7:
                    fenString += "p"
                elif tile == 2 or tile == 8:
                    fenString += "r"
                elif tile == 3 or tile == 9:
                    fenString += "n"
                elif tile == 4 or tile == 10:
                    fenString += "b"
                elif tile == 5 or tile == 11:
                    fenString += "q"
                elif tile == 6 or tile == 12:
                    fenString += "k"
                if tile <= 6 and tile >= 1:
                    fenString = fenString[:len(fenString)-1] + fenString[len(fenString)-1].upper()
            if y == 7:
                if isOpen == True:
                    isOpen = False
                    fenString += str(openSpaces)
                    openSpaces = 0
                if x != 7:
                    fenString += '/'
                else:
                    print("http://en.lichess.org/editor/%s" % fenString) #for determining if the FEN is correct
                    fenString += ' '
    fenString += turn + ' ' #this satisfies the second field

    wCastle = ""
    bCastle = ""
    if (boardArray[7][4] != 6) or (boardArray[7][0] != 2 and boardArray[7][7] != 2): #checking to see if castling
        wCastle = ""                                                             #is available for white
    elif boardArray[7][0] != 2 and boardArray[7][7] == 2:
        wCastle = "K"
    elif boardArray[7][0] == 2 and boardArray[7][7] != 2:
        wCastle = "Q"
    else:
        wCastle = "KQ"
    if (boardArray[0][4] != 12) or (boardArray[0][0] != 8 and boardArray[0][7] != 8): #checking to see if castling
        bCastle = ""                                                              #is available for black
    elif boardArray[0][0] != 8 and boardArray[0][7] == 8:
        bCastle = "k"
    elif boardArray[0][0] == 8 and boardArray[0][7] != 8:
        bCastle = "q"
    else:
        bCastle = "kq"
    if wCastle == bCastle:
        fenString += '-'
    else:
        fenString += wCastle + bCastle

    #determined that the last 3 fields are not necessary for Stockfish to still make its prediction
    #fenString += " - 0 1" #handles last 3 fields, but we aren't caring about them much right now

    #below, use your own path for your own stockfish install
    stockfish = Stockfish(path="/Users/david/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe", depth=18, parameters={"Threads": 4, "Minimum Thinking Time": 30})
    stockfish.update_engine_parameters({"Hash": 2048, "UCI_Chess960": "true"})
    stockfish.set_skill_level(20)
    stockfish.set_fen_position(fenString)
    bestMove =  stockfish.get_best_move()
    return bestMove

# This is just test code to make sure stockfish is working properly
#stockfish = Stockfish(path="/Users/david/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe", depth=18, parameters={"Threads": 2, "Minimum Thinking Time": 30})
#stockfish.update_engine_parameters({"Hash": 2048, "UCI_Chess960": "true"})
#stockfish.set_position(["e2e4", "e7e6"])
#move = stockfish.get_best_move()
#print(move) #should print d2d4

#this array represents the starting position of the chess
#exampleArray = [[8,9,10,11,12,10,9,8],[7,7,7,7,7,7,7,7],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1],[2,3,4,5,6,4,3,2]]
#outp = generateMove(exampleArray, 'W')
#print(outp)#should print e2e4