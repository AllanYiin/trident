import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from time import sleep

UNIT = 30  # Number of pixels
dy = [UNIT, 0, 0]  # Down, left, right
dx = [0, UNIT, -UNIT]

basic_counter_str = 'Test : '
basic_score_str = 'Score : '
basic_reward_str = 'Reward : '

NO_REWARD = -1000000

# w1 = 0.25
w1 = 1.0

w3 = 0.2

class Tetris(tk.Tk):
    def __init__(self, is_rendering=False, game_level=1, height=20, width=8, down_period=3):
        super(Tetris, self).__init__()

        self.height = height  # vertical grid
        self.width = width  # horizontal grid
        self.mid = (int(self.width / 2) - 1) * UNIT  # Block starting point
        self.down_period = down_period
        self.action_num = 0

        self.pos = [
            # O tetromino
            [[[self.mid, 0], [self.mid + UNIT, 0], [self.mid, 0 + UNIT], [self.mid + UNIT, 0 + UNIT]]],
            # I tetromino
            [[[self.mid - UNIT * 2, 0], [self.mid - UNIT, 0], [self.mid, 0], [self.mid + UNIT, 0]],
             [[self.mid, UNIT * 2], [self.mid, UNIT], [self.mid, 0], [self.mid, -UNIT]]],
            #L tetromino
            [[[self.mid, -UNIT], [self.mid, 0], [self.mid, UNIT], [self.mid + UNIT, UNIT]],
             [[self.mid + UNIT, 0], [self.mid, 0], [self.mid - UNIT, 0], [self.mid - UNIT, UNIT]],
             [[self.mid, UNIT], [self.mid, 0], [self.mid, -UNIT], [self.mid - UNIT, -UNIT]],
             [[self.mid - UNIT, 0], [self.mid, 0], [self.mid + UNIT, 0], [self.mid + UNIT, -UNIT]]],
            #J tetromino
            [[[self.mid, -UNIT], [self.mid, 0], [self.mid, UNIT], [self.mid - UNIT, UNIT]],
             [[self.mid + UNIT, 0], [self.mid, 0], [self.mid - UNIT, 0], [self.mid - UNIT, -UNIT]],
             [[self.mid, UNIT], [self.mid, 0], [self.mid, -UNIT], [self.mid + UNIT, -UNIT]],
             [[self.mid - UNIT, 0], [self.mid, 0], [self.mid + UNIT, 0], [self.mid + UNIT, UNIT]]],
            # Z tetromino
            [[[self.mid, UNIT], [self.mid, 0], [self.mid + UNIT, 0], [self.mid + UNIT, -UNIT]],
             [[self.mid - UNIT, 0], [self.mid, 0], [self.mid, UNIT], [self.mid + UNIT, UNIT]]],
            # S tetromino
            [[[self.mid, -UNIT], [self.mid, 0], [self.mid + UNIT, 0], [self.mid + UNIT, UNIT]],
             [[self.mid + UNIT, 0], [self.mid, 0], [self.mid, UNIT], [self.mid - UNIT, UNIT]]],
            # T tetromino
            [[[self.mid - UNIT, 0], [self.mid, 0], [self.mid + UNIT, 0], [self.mid, UNIT]],
             [[self.mid, -UNIT], [self.mid, 0], [self.mid, UNIT], [self.mid - UNIT, 0]],
             [[self.mid + UNIT, 0], [self.mid, 0], [self.mid - UNIT, 0], [self.mid, -UNIT]],
             [[self.mid, UNIT], [self.mid, 0], [self.mid, -UNIT], [self.mid + UNIT, 0]]],
        ]

        self.score = 0.0
        self.counter = 0

        self.color = ['white', 'magenta', 'red', 'green', 'blue', 'cyan', 'yellow']
        self.block_kind = len(self.pos)
        self.block = list()

        self.curr_block = np.random.randint(self.block_kind)
        self.curr_block_type = np.random.randint(len(self.pos[self.curr_block]))
        self.canvas, self.counter_board, self.score_board, self.reward_board = self._build_canvas()
        self.map = [[0] * self.width for _ in range(self.height)]
        self.shadow_map = [[0] * self.width for _ in range(self.height)]

        self.rendering = is_rendering  # True is rendering mode & False is no-rendering mode
        self.level = game_level  #Setting the difficulty of the game
        self.block_count = 1
        self.stack_height = 0

    def set_rendering(self, is_rendering):
        """ specify rendering on/off """
        self.rendering = is_rendering

    def set_game_level(self, game_level):
        """ Setting the difficulty of the game """
        self.level = game_level

    def _get_curr_block_pos(self):
        ret = []
        for n in range(4):
            s = (self.canvas.coords(self.block[n]))
            y = int(s[1] / UNIT)
            x = int(s[0] / UNIT)
            ret.append([y, x])
        return ret

    def _clear_map(self):
        for n in range(self.height):
            for m in range(self.width):
                self.map[n][m] = 0

    def _erase_down_canvas(self, iy):
        """ Remove all blocks in line iy """
        for crect in self.canvas.find_withtag("rect"):
            if int(self.canvas.coords(crect)[1]) == iy * UNIT:
                self.canvas.delete(crect)

    def _move_all_canvas_down(self, iy):
        """ Move all blocks above line iy down one space  """
        for crect in self.canvas.find_withtag("rect"):
            if int(self.canvas.coords(crect)[1]) < iy * UNIT:
                self.canvas.move(crect, 0, UNIT)

    def _move_all_canvas_up(self, num):
        for crect in self.canvas.find_withtag("rect"):
            if int(self.canvas.coords(crect)[1]) < num * UNIT:
                self.canvas.delete(crect)
            else:
                self.canvas.move(crect, 0, -num * UNIT)
        for m in range(self.width):
            for n in range(num):
                self.map[n][m] = 0
            for n in range(num, self.height, 1):
                self.map[n - num][m] = self.map[n][m]

    def _create_random_block(self, num, prob=0.5):
        a = np.random.rand(num, self.width)
        for m in range(self.width):
            for n in range(num):
                if (a[n, m] < prob):
                    self.map[self.height - 1 - n][m] = 1
                    self.canvas.create_rectangle(UNIT * m, UNIT * (self.height - 1 - n), UNIT * (m + 1),
                                                 UNIT * (self.height - n), fill='dim gray', tag="rect")
                else:
                    self.map[self.height - 1 - n][m] = 0
        self.canvas.pack()

    def shadow_position(self, ori=False):
        shadow = np.zeros((4, 2), dtype=int)
        if ori:
            origin = np.zeros((4, 2), dtype=int)
        for n in range(4):
            s = (self.canvas.coords(self.block[n]))
            if len(s):
                x = int(s[0] / UNIT)
                y = int(s[1] / UNIT)

                shadow[n, :] = x, y
        if ori:
            origin[:, :] = shadow[:, :]
        while 1:
            stop = False
            for n in range(4):
                x, y = shadow[n, :]
                if (y == self.height - 1) or (self.map[y + 1][x] == 1):
                    stop = True
            if stop:
                break
            shadow[:, 1] += 1
        if ori:
            return shadow, origin
        else:
            return shadow

    def _create_shadow(self):
        for crect in self.canvas.find_withtag("shadow"):
            self.canvas.delete(crect)
        shadow, origin = self.shadow_position(True)
        for n in range(4):
            x, y = shadow[n, :]
            hmmm = False
            for i in range(4):
                if (x == origin[i, 0]) & (y == origin[i, 1]):
                    hmmm = True
            if not hmmm:
                self.canvas.create_rectangle(UNIT * x, UNIT * y, UNIT * (x + 1),
                                             UNIT * (y + 1), fill='dim gray', tag="shadow")
        return shadow

    def calculate_reward(self):
        # reward_map = np.zeros((self.height, self.width), dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                self.shadow_map[i][j] = self.map[i][j]
        block_position = self.shadow_position()

        for i in range(4):
            x, y = block_position[i, :]
            self.shadow_map[y][x] = 1

        # blank = 0
        # for i in range(4):
        #     x, y = block_position[i,:]
        #     y += 1
        #     while y < self.height and self.shadow_map[y][x] == 0:
        #         blank+=1
        #         y += 1

        break_cnt = 0
        for n in range(self.height - 1, 0, -1):
            cnt = 0
            for m in range(self.width):
                if self.shadow_map[n][m] != 1:
                    break
                cnt += 1
            if cnt == self.width:
                break_cnt += 1

        return break_cnt * w1 * break_cnt * break_cnt # + blank*w3

    def _add_canvas(self):
        """ 새로운 block생성 후 canvas에 추가 """
        pos = self.make_block()
        rect1 = self.canvas.create_rectangle(pos[0][0], pos[0][1], pos[0][0] + UNIT,
                                             pos[0][1] + UNIT, fill=self.color[self.curr_block],
                                             tag="rect")
        rect2 = self.canvas.create_rectangle(pos[1][0], pos[1][1], pos[1][0] + UNIT,
                                             pos[1][1] + UNIT, fill=self.color[self.curr_block],
                                             tag="rect")
        rect3 = self.canvas.create_rectangle(pos[2][0], pos[2][1], pos[2][0] + UNIT,
                                             pos[2][1] + UNIT, fill=self.color[self.curr_block],
                                             tag="rect")
        rect4 = self.canvas.create_rectangle(pos[3][0], pos[3][1], pos[3][0] + UNIT,
                                             pos[3][1] + UNIT, fill=self.color[self.curr_block],
                                             tag="rect")

        self.block = [rect1, rect2, rect3, rect4]
        self.canvas.pack()

    def _build_canvas(self):
        """ Initializing the canvas for the first time """
        canvas = tk.Canvas(self, bg='black',
                           height=self.height * UNIT,
                           width=(self.width + 7) * UNIT)

        # Score background plate
        base_point = UNIT * self.width
        canvas.create_rectangle(35 + base_point, 150, 185 + base_point, 275, fill='dim gray')
        canvas.create_text(105 + base_point, UNIT * self.height - 10,
                           font="Times 11 bold",
                           fill='white',
                           text="Tetris")

        counter_board = canvas.create_text(110 + base_point, 175,
                                           fill="gray22",
                                           font="Times 13 bold",
                                           text=basic_counter_str + str(int(self.counter)))

        score_board = canvas.create_text(110 + base_point, 205,
                                         fill="gray22",
                                         font="Times 13 bold",
                                         text=basic_score_str + str(int(self.score)))

        reward_board = canvas.create_text(110 + base_point, 230,
                                          fill="gray22",
                                          font="Times 13 bold",
                                          text=basic_reward_str + str(0))

        # Grid creation
        for c in range(0, (self.width + 1) * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, self.height * UNIT
            canvas.create_line(x0, y0, x1, y1, fill='white')

        for r in range(0, self.height * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, self.width * UNIT, r
            canvas.create_line(x0, y0, x1, y1, fill='white')

        # Add an image to the canvas
        pos = self.make_block()
        rect1 = canvas.create_rectangle(pos[0][0], pos[0][1], pos[0][0] + UNIT,
                                        pos[0][1] + UNIT, fill=self.color[self.curr_block],
                                        tag="rect")
        rect2 = canvas.create_rectangle(pos[1][0], pos[1][1], pos[1][0] + UNIT,
                                        pos[1][1] + UNIT, fill=self.color[self.curr_block],
                                        tag="rect")
        rect3 = canvas.create_rectangle(pos[2][0], pos[2][1], pos[2][0] + UNIT,
                                        pos[2][1] + UNIT, fill=self.color[self.curr_block],
                                        tag="rect")
        rect4 = canvas.create_rectangle(pos[3][0], pos[3][1], pos[3][0] + UNIT,
                                        pos[3][1] + UNIT, fill=self.color[self.curr_block],
                                        tag="rect")
        self.block = [rect1, rect2, rect3, rect4]
        canvas.pack()
        return canvas, counter_board, score_board, reward_board

    def get_height(self):
        for n in range(self.height):
            for m in range(self.width):
                if self.map[n][m] == 1:
                    return n
        return self.height

    def make_block(self):
        """ 현재 block의 모양 정보 return """
        return self.pos[self.curr_block][self.curr_block_type]

    def reset(self):
        """ episode가 끝난 후 새 게임을 위한 reset """
        self.score = 0.0
        self.counter += 1
        self.canvas.itemconfigure(self.counter_board,
                                  text=basic_counter_str + str(int(self.counter)))
        self.canvas.itemconfigure(self.score_board,
                                  text=basic_score_str + str(int(self.score)))
        self.canvas.itemconfigure(self.reward_board,
                                  text=basic_reward_str + str(0))
        if self.rendering:
            self.update()
        self.canvas.delete("rect")
        self._clear_map()
        self._add_canvas()
        self.block_count = 1
        self.action_num = 0

    def step(self, action):
        """ Game progress according to action  """
        if self.rendering:
            self.render()

        self.action_num = (self.action_num + 1) % self.down_period

        reward = float

        if action < 3:
            # 0 Down 1 left 2 right
            reward = self.move(action)
        elif action < 5:
            # 3 is clockwise 4 is counterclockwise
            reward = self.rotate(action)
        else:
            while True:
                reward = self.move(0)
                if reward != NO_REWARD:
                    break
        if self.action_num == 0 and reward == NO_REWARD:
            reward = self.move(0)

        #self._create_shadow()

        if reward != NO_REWARD:
            # make new block!
            self.curr_block = np.random.randint(self.block_kind)
            self.curr_block_type = np.random.randint(len(self.pos[self.curr_block]))
            self._add_canvas()
            self.action_num = 0
            #self._create_shadow()

            # if reward == 0:
            #     height = self.height - self.get_height()
            #     reward = (height / self.height) * (-0.5)

        shadow_reward = self.calculate_reward()
        stack_height = self.height - self.stack_height
        if self.is_game_end():
            reward = -1
            is_end = True
        else:
            is_end = False
        if reward == NO_REWARD:
            return 0.0, shadow_reward, is_end, False, 0
        else:
            return reward, shadow_reward, is_end, True, stack_height

    def possible_to_move(self, action):
        """ Make sure you can move
        return 1  If there is a block in the position you want to move out of or move
        return 2 When moving down When it reaches the bottom or there is a block below it
        return 3 Moveable """
        for n in range(len(self.block)):
            s = self.canvas.coords(self.block[n])
            y = s[1] + dy[action]
            x = s[0] + dx[action]

            # Out of range  - stay
            if x >= self.width * UNIT or x < 0:
                return 1
            ny = int(y / UNIT)
            nx = int(x / UNIT)

            # Last line  - add canvas
            if y >= self.height * UNIT:
                return 2
            if self.map[ny][nx] == 1:
                if action == 0:
                    return 2
                else:
                    return 1

        # Moveable  - move
        return 3

    def is_map_horizon(self):
        """ Check for full horizontal lines
        return -1 No full line
        return n Returns the lowest index among full lines
        """
        for n in range(self.height - 1, 0, -1):
            cnt = 0
            for m in range(self.width):
                if self.map[n][m] != 1:
                    break
                cnt += 1
            if cnt == self.width:
                return n
        return -1

    def move(self, action):
        """ action 0 moves down, 1 moves left, 2 moves right  """
        # ret은 reward
        ret = 0.0

        # flag 1 There is a block off the map or where you want to move
        # flag 2 When moving down When it reaches the bottom or there is a block below it
        # flag 3 Moveable
        flag = self.possible_to_move(action)

        if flag == 2:  # The block reaches the bottom and ends the movement.
            self.block_count += 1
            #Show current block on map
            self.stack_height = 0
            for n in range(4):
                s = (self.canvas.coords(self.block[n]))
                y = int(s[1] / UNIT)
                x = int(s[0] / UNIT)
                self.map[y][x] = 1
                if y > self.stack_height:
                    self.stack_height = y

            # If the horizontal line is full, empty it and add the score.
            break_cnt = 0
            while True:
                y = self.is_map_horizon()
                if y == -1:
                    break
                self._erase_down_canvas(y)
                self._move_all_canvas_down(y)
                break_cnt += 1
                for m in range(self.width):
                    for n in range(y, 2, -1):
                        self.map[n][m] = self.map[n - 1][m]
            self.score += w1 * break_cnt * break_cnt * break_cnt
            ret += w1 * break_cnt * break_cnt * break_cnt


            # if break_cnt == 0:
            #     blank = 0
            #     for n in range(4):
            #         s = (self.canvas.coords(self.block[n]))
            #         y = int(s[1] / UNIT) + 1
            #         x = int(s[0] / UNIT)
            #
            #         while y < self.height and self.map[y][x] == 0:
            #             blank+=1
            #             y += 1
            #     ret -= w3 * blank
            #     self.score -= w3 * blank
            if not self.block_count % 10:
                if self.level == 2:
                    self._move_all_canvas_up(1)
                    self._create_random_block(1)
                elif self.level == 3:
                    self._move_all_canvas_up(3)
                    self._create_random_block(3)

            self.canvas.itemconfigure(self.score_board,
                                      text=basic_score_str + str(int(self.score)))
            self.canvas.itemconfigure(self.reward_board,
                                      text=basic_reward_str + str(int(ret)))
            return ret

        # move
        elif flag == 3:
            for n in range(4):
                self.canvas.move(self.block[n], dx[action], dy[action])

        return NO_REWARD

    def possible_to_rotate(self, next_block):
        """ Check if there is another block at the position of next_block or out of the map  """
        for i in range(len(self.block)):
            y = int(next_block[i][1] / UNIT)
            x = int(next_block[i][0] / UNIT)
            if y < 0 or y >= self.height or x < 0 or x >= self.width or self.map[y][x] == 1:
                return False
        return True

    def rotate(self, dir):
        """ direction 3 is clockwise, 4 is counterclockwise rotation """
        dir = (1 if dir == 3 else 3)
        next_block = [[0] * 2 for _ in range(len(self.block))]
        curr_size = len(self.pos[self.curr_block])
        for i in range(len(self.block)):
            s = self.canvas.coords(self.block[i])
            # y
            next_block[i][1] = s[1] + self.pos[self.curr_block][(self.curr_block_type + dir) % curr_size][i][1] - \
                               self.pos[self.curr_block][self.curr_block_type][i][1]
            # x
            next_block[i][0] = s[0] + self.pos[self.curr_block][(self.curr_block_type + dir) % curr_size][i][0] - \
                               self.pos[self.curr_block][self.curr_block_type][i][0]

        if not self.possible_to_rotate(next_block):
            return NO_REWARD

        for i in range(len(self.block)):
            self.canvas.move(self.block[i],
                             self.pos[self.curr_block][(self.curr_block_type + dir) % curr_size][i][0] -
                             self.pos[self.curr_block][self.curr_block_type][i][0],
                             self.pos[self.curr_block][(self.curr_block_type + dir) % curr_size][i][1] -
                             self.pos[self.curr_block][self.curr_block_type][i][1])
        self.curr_block_type = (self.curr_block_type + dir) % curr_size
        return NO_REWARD

    def is_game_end(self):
        for n in range(3):
            for m in range(self.width):
                if self.map[n][m] == 1:
                    return True
        return False

    def render(self):
        #render visualization and setting frames rate
        sleep(0.01)
        self.update()