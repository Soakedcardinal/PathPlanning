import sys
import math
import numpy as np
import random
import operator

'''
A* Pathfinding algorithm 

@author: Duncan Nicholson

HOW TO RUN THIS CODE: 
    
    $ python3 Astar.py

Created with Python 3.7.7
'''

# PARAMETERS
#############
SHOW_ITERATIONS = False
MAX_ITS = 1e4
#############
BOARD_SIZE = [11, 11]
START = [0, 0]
GOAL = [10, 10]
TREES = [[1, 5], [1, 6],
         [2, 5], [2, 6], [2, 7],
         [3, 6], [3, 7], [3, 8],
         [4, 5], [4, 6], [4, 7], [4, 8],
         [5, 4], [5, 5], [5, 6],
         [6, 3], [6, 4], [6, 5],
         [7, 3], [7, 4]]
#############
DO_RANDOM_BOARDS = True


class Tile:
    def __init__(self, r, c, _type):
        # location
        self.r = r
        self.c = c

        # string for storing the type, " "=EMPTY, *=TREE, $=START,#=GOAL
        self.type = _type

        self.visited = False
        self.current = False
        self.cost = float("inf")
        self.heuristic = 0

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
            self.tiles[tree[0]][tree[1]].cost = float("inf")

        # set start and end locations
        self.tiles[self.start[0]][self.start[1]].type = "$"
        self.tiles[self.goal[0]][self.goal[1]].type = "#"

        # set the start node as current and set it's distance to 0
        self.tiles[self.start[0]][self.start[1]].current = True
        self.tiles[self.start[0]][self.start[1]].cost = 0

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
    def getNeighbors(self):
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
                if tile.cost == float("inf"):
                    sys.stdout.write(f"|  {tile.cost}")
                else:
                    sys.stdout.write("|%5.1f" % tile.cost)
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


class Astar:
    def __init__(self, _board):
        print("getting ready for Astar algorithm . . .")

        self.nits = 0

        # shallow copy the board
        self.board = _board

        # show the starting board
        self.board.print_board([])

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

            # stop if we've reached the destination
            if tile.r == self.board.goal[0] and tile.c == self.board.goal[1]:
                print("stopping: destination reached")
                break

            costs = []

            nbs = self.board.getNeighbors()
            for nb in nbs:
                if nb.type == " " or nb.type == "#":
                    if nb.visited == False:
                        if nb.r == tile.r or nb.c == tile.c:
                            # neighbor is adjacent
                            nb_dist = 1.0
                        else:
                            # neighbor is diagonal: distance is hypotenouse of 45-45-90 triangle
                            nb_dist = math.sqrt(2.0)

                        # ADDITION FOR A* ########################################################
                        nb.heuristic = np.sqrt((self.board.goal[0] - nb.r)**2 + (self.board.goal[1] - nb.c)**2)
                        # alternatively
                        # heuristic = (self.board.goal[0] - nb.r) + (self.board.goal[1] - nb.c)

                        nb.cost = tile.cost + nb_dist + nb.heuristic

                        costs.append((nb.cost, nb))

                        # keep track of shortest path
                        nb.prev = [tile.r, tile.c]

            sorted_costs = sorted(costs, key=lambda x: x[0])

            next_tile = sorted_costs[0][1]

            next_tile.current = True
            tile.current = False
            tile.visited = True

            if SHOW_ITERATIONS:
                print("\ncurrent node: [{}, {}]".format(tile.r, tile.c))
                sys.stdout.write("neighbors: ")
                for tup in sorted_costs:
                    sys.stdout.write(f"{[tup[1].r,tup[1].c]} ")
                sys.stdout.write("\nwith costs: ")
                for tup in sorted_costs:
                    sys.stdout.write(f"{tup[0]:.4f} ")
                sys.stdout.write("\n")
                print(f"minimum cost: {sorted_costs[0][0]:.4f}")
                print("selecting node: [{}, {}]".format(next_tile.r, next_tile.c))
                self.board.print_distances()

            self.nits += 1
            if self.nits > MAX_ITS:
                print("stopping: couldn't reach destination in {} iterations".format(MAX_ITS))
                break

    def recover_path(self):

        #  start at the destination
        location = self.board.goal

        thepath = [location]

        # # walk back along the shortest path
        while location != self.board.start:
            # get the tile object at that location
            for row in self.board.tiles:
                for tile in row:
                    if [tile.r, tile.c] == location:
                        thetile = tile

            location = thetile.prev
            thepath.append(location)

        # reverse the path
        for node in thepath:
            node = node[::-1]

        return thepath[::-1]

    def print_text_path(self, __path):
        print("\nshortest path:")
        for _loc in __path:
            sys.stdout.write(str(_loc))
            if __path.index(_loc) != len(__path)-1:
                 sys.stdout.write("->")


def locate_min(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a)
                      if smallest == element]


def main():
    # instantiate a board object
    b = Board(BOARD_SIZE, START, GOAL, TREES)

    # run Astar's algorithm on that board
    Astar(b)

    # easter egg?
    if DO_RANDOM_BOARDS:
        tree_density = 0.5
        nboards = 10
        for _ in range(nboards):
            size2 = [random.randint(15,20) for _ in range(2)]
            print(size2)

            start2 = [0,0] #[random.randint(0,size2[0]-1),random.randint(0,size2[1]-1)]
            goal2 = [size2[0]-1,size2[1]-1]# [random.randint(0,size2[0]-1),random.randint(0,size2[1]-1)]
            # if goal2 == start2:
            #     goal2 = [random.randint(0,size2[0]-1),random.randint(0,size2[1]-1)]
            random_trees = [[random.randint(0,size2[0]-1),random.randint(0,size2[1]-1)] for _ in range(random.randint(0,int(size2[0]*size2[1]*tree_density)))]
            if start2 in random_trees: random_trees.remove(start2)
            if goal2 in random_trees: random_trees.remove(goal2)

            Astar(Board(size2,start2,goal2,random_trees))


if __name__ == "__main__":
    main()
