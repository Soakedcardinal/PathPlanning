import sys
import math
import numpy as np
import random

'''
Dijkstras algorithm 

@author: Duncan Nicholson

HOW TO RUN THIS CODE: 
    
    $ python Dijkstra.py

if you don't have numpy installed

    $ pip install numpy

Tested with Python 3.7.7
'''

# PARAMETERS
#############
SHOW_ITERATIONS = False
MAX_ITS = 1e4
#############
BOARD_SIZE = [11, 11]
START = [0, 0]
GOAL = [10, 10]
TREES = [[2, 2], [2, 3], [3, 3], [3, 4], [3, 5],
         [4, 5], [4, 6], [4, 7], [5, 7], [5, 8]]
#############


class Tile:
    def __init__(self, r, c, _type):
        # location
        self.r = r
        self.c = c

        # string for storing the type, " "=EMPTY, *=TREE, $=START,#=GOAL
        self.type = _type

        self.visited = False
        self.current = False
        self.distance = float("inf")

        ''' 
         hold previous tile that "discovered" the current tile
        we will iterate again after the algorithm is over to 
        to recover the shortest path 
        '''
        self.prev = []


class Board:
    def __init__(self, _size, _start, _goal, _trees):
        print("constructing board ...")
        self.size = _size
        self.start = _start
        self.goal = _goal
        self.trees = _trees
        self.nrows = _size[0]
        self.ncols = _size[1]

        # initialize empty board as a list of lists
        self.tiles = [[] for _ in range(self.nrows)]

        # fill the board with empty tiles
        for r in range(self.nrows):
            for c in range(self.ncols):
                self.tiles[r].append(Tile(r, c, _type=" "))

        # set tiles that are obstacles
        for tree in _trees:
            self.tiles[tree[0]][tree[1]].type = "*"
            self.tiles[tree[0]][tree[1]].distance = float("inf")

        # set start and end locations
        self.tiles[self.start[0]][self.start[1]].type = "$"
        self.tiles[self.goal[0]][self.goal[1]].type = "#"

        # set the start node as current and set it's distance to 0
        self.tiles[self.start[0]][self.start[1]].current = True
        self.tiles[self.start[0]][self.start[1]].distance = 0

    # return current tile object
    def getCurrentTile(self):
        for row in self.tiles:
            for __tile in row:
                if __tile.current == True:
                    return __tile

    # return all unvisited tiles
    def getUnvisited(self):
        unv_tiles = []
        for row in self.tiles:
            for tile in row:
                if tile.visited == False and tile.type != "*":
                    unv_tiles.append(tile)
        return unv_tiles

    # return unvisited tiles next to current tile
    def getUnvisitedNeighbors(self):
        tile = self.getCurrentTile()
        r = tile.r
        c = tile.c
        ts = self.tiles
        maxr = self.size[0]-1
        maxc = self.size[1]-1

        if r == 0:  # no SE S SW
            if c == 0:  # No NW W SW
                nbs = [ts[r+1][c],  # N
                       ts[r+1][c+1],  # NE
                       ts[r][c+1]]  # E
            elif c == maxc:  # no NE E SE
                nbs = [ts[r+1][c],  # N
                       ts[r][c-1],  # W
                       ts[r+1][c-1]]  # NW
            else:
                nbs = [ts[r+1][c],  # N
                       ts[r+1][c+1],  # NE
                       ts[r][c+1],  # E
                       ts[r][c-1],  # W
                       ts[r+1][c-1]]  # NW
        elif r == maxr:  # no N NE NW
            if c == 0:  # No NW W SW
                nbs = [ts[r][c+1],  # E
                       ts[r-1][c+1],  # SE
                       ts[r-1][c]]  # S
            elif c == maxc:  # no NE E SE
                nbs = [ts[r-1][c],  # S
                       ts[r-1][c-1],  # SW
                       ts[r][c-1]]  # W
            else:
                nbs = [ts[r][c+1],  # E
                       ts[r-1][c+1],  # SE
                       ts[r-1][c],  # S
                       ts[r-1][c-1],  # SW
                       ts[r][c-1]]  # W
        elif c == 0:  # no NW W SW
            nbs = [ts[r+1][c],  # N
                   ts[r+1][c+1],  # NE
                   ts[r][c+1],  # E
                   ts[r-1][c+1],  # SE
                   ts[r-1][c]]  # S
        elif c == maxc:  # no NE E SE
            nbs = [ts[r+1][c],  # N
                   ts[r-1][c],  # S
                   ts[r-1][c-1],  # SW
                   ts[r][c-1],  # W
                   ts[r+1][c-1]]  # NW
        else:
            nbs = [ts[r+1][c],  # N
                   ts[r+1][c+1],  # NE
                   ts[r][c+1],  # E
                   ts[r-1][c+1],  # SE
                   ts[r-1][c],  # S
                   ts[r-1][c-1],  # SW
                   ts[r][c-1],  # W
                   ts[r+1][c-1]]  # NW

        # trim trees and visited tiles from neighbors
        for nb in nbs:
            if nb.visited == True or nb.type == "*":
                nbs.remove(nb)

        return nbs

    # show the board to the user, and optionally the path
    def print_board(self, _path):
        # top line
        sys.stdout.write("\n   ")
        for _ in range(self.ncols):
            sys.stdout.write("--")
        print()
        for row in self.tiles[::-1]:
            # row legend
            if row[0].r < 10:
                sys.stdout.write(" ")
            sys.stdout.write(str(row[0].r)+" ")

            # board content
            for tile in row:
                if [tile.r, tile.c] in _path:
                    sys.stdout.write("|@")
                else:
                    sys.stdout.write("|"+tile.type)

            sys.stdout.write("|\n")
        # bottom line
        sys.stdout.write("   ")
        for _ in range(self.ncols):
            sys.stdout.write("--")
        print()
        # column legend
        sys.stdout.write("    ")
        for num in range(self.ncols):
            sys.stdout.write(str(num)+" ")
        print()

    def print_distances(self):
        # top line
        sys.stdout.write("\n   ")
        for _ in range(self.ncols):
            sys.stdout.write("--")
        print()

        for row in self.tiles[::-1]:
            # row legend
            if row[0].r < 10:
                sys.stdout.write(" ")
            sys.stdout.write(str(row[0].r)+" ")

            # board content
            for tile in row:
                if tile.distance == float("inf"):
                    sys.stdout.write(f"| {tile.distance}")
                elif tile.distance < 10:
                    sys.stdout.write("| %3.1f" % tile.distance)
                else:
                    sys.stdout.write("|%3.1f" % tile.distance)
            sys.stdout.write("|\n")

        # bottom line
        sys.stdout.write("   ")
        for _ in range(self.ncols):
            sys.stdout.write("--")
        print()

        # column legend
        sys.stdout.write("    ")
        for num in range(self.ncols):
            sys.stdout.write(str(num)+" ")
        print()


class Dijkstra:
    def __init__(self, _board):
        print("getting ready for dijkstras algorithm . . .")

        self.nits = 0

        # shallow copy the board
        self.board = _board

        # show the starting board
        self.board.print_board([])

        # instantiate unvisited set as a data member
        self.unvisited = self.board.getUnvisited()

        # run Djkstras algorithm
        self.iterate()

        # print results
        if self.nits < MAX_ITS:

            path = self.recover_path()

            self.print_text_path(path)

            self.board.print_board(path)

    def iterate(self):
        print("iterating . . .")
        while True:
            tile = self.board.getCurrentTile()

            # for debug printout only
            nb_dists = []

            thru_dists = []
            nbs = self.board.getUnvisitedNeighbors()
            for nb in nbs:
                if nb.r == tile.r or nb.c == tile.c:
                    # neighbor is adjacent
                    dist = 1.0
                else:
                    # neighbor is diagonal
                    # immediate distance is hypotenouse of 45-45-90 triangle
                    dist = math.sqrt(2.0)
                nb_dists.append(dist)

                # store distances to unvisited neighbors THRU the current node
                thru_dists.append(tile.distance + dist)

             # if the smallest neighbor distance is infinity (eg stuck inside obstacles), then stop
            if min(nb_dists) == float("inf"):
                print("stopping: minimum neighbor distance is inf")
                break

            # if the new tentative distance is lower than currently held tentative distance for that nb, update it
            for ind, nb in enumerate(nbs):
                if thru_dists[ind] < nb.distance and nb.type == " ":
                    nb.distance = thru_dists[ind]

                    # keep track of shortest path to that tile
                    nb.prev = [tile.r, tile.c]

            tile.visited = True
            self.unvisited.remove(tile)

            # stop if we've visited the destination
            if self.unvisited == []:
                self.board.tiles[10][10].prev = [9, 9]
                print("stopping: destination reached")
                break

            # loop over ALL unvisited open tiles and store the distances
            unv_thru_dists = []
            for unv_tile in self.unvisited:
                unv_thru_dists.append(unv_tile.distance)

            # select the next tile with lowest tentative distance of unvisited tiles that aren't trees
            min_thru_dist = min(unv_thru_dists)
            next_tile_ind = unv_thru_dists.index(min_thru_dist)
            next_tile = self.unvisited[next_tile_ind]
            next_tile.current = True
            tile.current = False

            if SHOW_ITERATIONS:
                nb_locs = []
                for nb in nbs:
                    nb_locs.append([nb.r, nb.c])
                print("\ncurrent node: [{}, {}]".format(tile.r, tile.c))
                print("{} neighbors: {}".format(len(nb_locs), nb_locs))
                print("with distances: {}".format(
                    ['%.2f' % elem for elem in nb_dists]))
                print("with total distances: {}".format(
                    ['%.2f' % elem for elem in thru_dists]))
                self.board.print_distances()

            self.nits += 1
            if self.nits > MAX_ITS:
                print(
                    "stopping: couldn't reach destination in {} iterations".format(MAX_ITS))
                break

    def recover_path(self):

        #  start at the destination
        locatio = self.board.goal

        thepath = [locatio]

        # # walk back along the shortest path
        while locatio != self.board.start:

            # get the tile object at that location
            for row in self.board.tiles:
                for tile in row:
                    if [tile.r, tile.c] == locatio:
                        thetile = tile

            locatio = thetile.prev
            thepath.append(locatio)

        # reverse the path
        return thepath[::-1]

    def print_text_path(self, __path):
        print("\nshortest path:")
        for _loc in __path:
            sys.stdout.write(str(_loc))
            if __path.index(_loc) != len(__path)-1:
                sys.stdout.write("->")


def main():
    # instantiate a board object
    b = Board(BOARD_SIZE, START, GOAL, TREES)

    # run Dijkstra's algorithm on that board
    _ = Dijkstra(b)

    # try some more complicated boards . .
    # tree_density = 0.2
    # nboards = 3
    # for _ in range(nboards):
    #     size2 = [random.randint(5,20) for _ in range(2)]
    #     start2 = [random.randint(0,size2[0]-1),random.randint(0,size2[1]-1)]
    #     goal2 = [random.randint(0,size2[0]-1),random.randint(0,size2[1]-1)]
    #     if goal2 == start2:
    #         goal2 = [random.randint(0,size2[0]-1),random.randint(0,size2[1]-1)]
    #     random_trees = [[random.randint(0,size2[0]-1),random.randint(0,size2[1]-1)] for _ in range(random.randint(0,int(size2[0]*size2[1]*tree_density)))]
    #     if start2 in random_trees: random_trees.remove(start2)
    #     if goal2 in random_trees: random_trees.remove(goal2)

    #     print("##################################################################")
    #     Dijkstra(Board(size2,start2,goal2,random_trees))


if __name__ == "__main__":
    main()
