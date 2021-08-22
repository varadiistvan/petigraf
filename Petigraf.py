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

turn=RED
chosen= ""
missing = []

vertices = [{"pos":(50,250),"color":BLACK, "key":0},{"pos":(125,50),"color":BLACK, "key":1},{"pos":(375,50),"color":BLACK, "key":2},{"pos":(450,250),"color":BLACK, "key":3},{"pos":(375,450),"color":BLACK, "key":4},{"pos":(125,450),"color":BLACK, "key":5}]
lines = []

def setup():
    pass

def draw_window():
    WIN.fill(WHITE)
    for vertex in vertices:
        pygame.draw.circle(WIN,vertex["color"],vertex["pos"],20)
    for line in lines:
        pygame.draw.line(WIN, line['color'], vertices[line['start']]['pos'], vertices[line['end']]['pos'], width=5)
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
                                vertex["color"] = turn
                            else:
                                if vertex["color"] != BLACK:
                                    chosen = ""
                                    vertex["color"] = BLACK
                                elif {'color': BLUE, 'start' : min(vertex["key"], chosen["key"]), 'end': max(vertex["key"], chosen["key"])} in lines or {'color': RED, 'start' : min(vertex["key"], chosen["key"]), 'end': max(vertex["key"], chosen["key"])} in lines:
                                    pass
                                else:
                                    for line in lines:
                                        if line['color'] == turn:
                                            if line['start'] == vertex["key"]:
                                                missing.append((turn, min(chosen['key'],line['end']), max(chosen['key'],line['end'])))
                                            if line['start'] == chosen["key"]:
                                                missing.append((turn, min(vertex['key'],line['end']), max(vertex['key'],line['end'])))
                                            if line['end'] == vertex["key"]:
                                                missing.append((turn, min(chosen['key'],line['start']), max(chosen['key'],line['start'])))
                                            if line['end'] == chosen["key"]:
                                                missing.append((turn, min(vertex['key'],line['start']), max(vertex['key'],line['start'])))
                                    lines.append({'color': turn, 'start' : min(vertex["key"], chosen["key"]), 'end': max(vertex["key"], chosen["key"])})
                                    opposite = ""
                                    if turn == BLUE:
                                        opposite = RED  
                                    else: opposite = BLUE
                                    if (opposite, min(vertex["key"], chosen["key"]), max(vertex["key"], chosen["key"])) in missing:
                                        missing.remove((opposite, min(vertex["key"], chosen["key"]), max(vertex["key"], chosen["key"])))
                                    vertex['color'] = BLACK
                                    vertices[chosen['key']]['color'] = BLACK
                                    if (turn, min(vertex["key"], chosen["key"]) , max(vertex["key"], chosen["key"])) in missing :
                                        draw_window()
                                        gameover()
                                        break
                                    if turn == RED:
                                        turn = BLUE
                                    else:
                                        turn = RED
                                    chosen = ""

        draw_window()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()