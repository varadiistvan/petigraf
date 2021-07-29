import math, pygame, sys


WIDTH = 500
HEIGHT = 500

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Vonalazzá")
pygame.init()

WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (0,0,255)
RED = (255,0,0)
FONT_IMPACT = pygame.font.SysFont("impact", 50)

FPS = 60

turn= 1
chosen= ""
missing = []
allLines = []
lines = []

turns = ["", RED, BLUE]


vertices = [{"pos":(50,250),"color":BLACK, "key":0},{"pos":(125,50),"color":BLACK, "key":1},{"pos":(375,50),"color":BLACK, "key":2},{"pos":(450,250),"color":BLACK, "key":3},{"pos":(375,450),"color":BLACK, "key":4},{"pos":(125,450),"color":BLACK, "key":5}]

for i in range(len(vertices)-1):
    for j in range(i+1, len(vertices)):
        allLines.append((i, j))
        lines.append(0)



def setup():
    pass

def draw_window():
    WIN.fill(WHITE)
    for vertex in vertices:
        pygame.draw.circle(WIN,vertex["color"],vertex["pos"],20)
    for i in range(len(lines)):
        if lines[i] != 0:
            pygame.draw.line(WIN, turns[lines[i]], vertices[allLines[i][0]]["pos"], vertices[allLines[i][1]]["pos"], width=5)
    pygame.display.update()

def gameover():
    draw_window()
    pygame.time.delay(1000)
    pygame.quit()
    sys.exit()

def main():
    global chosen, turn, missing
    clock=pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for vertex in vertices:
                        if event.pos[0] > vertex["pos"][0] - 20 and event.pos[0] < vertex["pos"][0] + 20 and event.pos[1] > vertex["pos"][1] - 20 and event.pos[1] < vertex["pos"][1] + 20:
                            if type(chosen) == str:
                                chosen = vertex
                                vertex["color"] = turns[turn]
                            else:
                                if vertex["color"] != BLACK:
                                    chosen = ""
                                    vertex["color"] = BLACK
                                elif lines[allLines.index((min(chosen["key"], vertex["key"]), max(chosen["key"], vertex["key"])))] != 0:
                                    pass
                                else:
                                    drawLine(min(vertex["key"], chosen["key"]), max(vertex["key"], chosen["key"]))
                                    vertex["color"] = BLACK
                                    vertices[chosen["key"]]["color"] = BLACK
                                    chosen = ""

        draw_window()

    pygame.quit()
    sys.exit()


def drawLine(start, end):
    global turn
    if(lines[allLines.index((start, end))] == 0):
        for i in range(len(lines)):
            if(lines[i] == turn):
                if start == allLines[i][0]:
                    missing.append((turn, min(end, allLines[i][1]), max(end, allLines[i][1])))
                if end == allLines[i][0]:
                    missing.append((turn, min(start, allLines[i][1], max(start, allLines[i][1]))))
                if start == allLines[i][1]:
                    missing.append((turn, min(end, allLines[i][0], max(end, allLines[i][0]))))
                if end == allLines[i][1]:
                    missing.append((turn, min(start, allLines[i][0]), max(start, allLines[i][0])))
        lines[allLines.index((start, end))] = turn
        if (-turn, start, end) in missing:
            missing.remove((-turn, start, end))
        if (turn, start , end) in missing :
            draw_window()
            gameover()
        else:
            turn = -turn





if __name__ == "__main__":
    main()