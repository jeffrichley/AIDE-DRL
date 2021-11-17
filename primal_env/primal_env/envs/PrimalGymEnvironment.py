import math
from math import sqrt
import itertools
# from tf_agents.specs import array_spec
# from tf_agents.environments import py_environment
import numpy as np
import gym
from gym.spaces import Box, Discrete

import primal_env.envs.TrainingFactory as TrainingFactory


FOUR_CONNECTED_MOVES = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]


# class BasePrimalGymEnvironment(py_environment.PyEnvironment):
class BasePrimalGymEnvironment(gym.Env):

    def get_info(self):
        return {}

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def __init__(self):

        self.window_x, self.window_y = 11, 11
        # self.window_x, self.window_y = 31, 31
        # self.window_x, self.window_y = 21, 21
        # self.window_x, self.window_y = 30, 30

        trainer, simulator, state, visualizer = TrainingFactory.create(config_file=self.get_config_file())

        self.trainer = trainer
        self.simulator = simulator
        self.state = state
        self.visualizer = visualizer

        self.round_count = 0

        # TODO: for now we are driving the 0th asset, need to work towards multi-agent
        self.asset_number = 0


        # how many actions do we have to learn?
        self.action_space = Discrete(5)
        # self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=5, name='action')

        # what the agent sees for observations
        # observation = self.reset()
        # y, x, _ = observation.shape
        x, y = self.window_x, self.window_y
        self.observation_space = Box(low=0.0, high=255.0, shape=(y, x, 3), dtype=np.float32)
        # self.observation_space = Box(low=0.0, high=255.0, shape=(11, 11, 3), dtype=np.float32)
        # self._observation_spec = array_spec.BoundedArraySpec(shape=(11, 11), dtype=np.float32, minimum=0.0, maximum=255.0, name='observation')



        # test = Box(low=0.0, high=30, shape=(4), dtype=np.float32)

        import gym.spaces as spaces
        self.observation_space = spaces.Dict({'image': Box(low=0.0, high=255.0, shape=(y, x, 3), dtype=np.float32),
                                              'location': Box(low=0.0, high=30, shape=(1, 4), dtype=np.float32)})



    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # def step(self, action):
    #     return self._step(action)

    # def _step(self, action):
    def step(self, action):
        self.round_count += 1

        asset = self.state.assets[self.asset_number]
        goal = self.state.goals[self.asset_number]

        move = FOUR_CONNECTED_MOVES[action]
        new_location = (asset[0] + move[0], asset[1] + move[1])

        moved, reward = self.simulator.move_asset(self.asset_number, new_location)

        # reward = self.trainer.get_reward(self.asset_number, self.state)
        done = new_location == goal or self.round_count >= 200
        # done = new_location == goal or self.round_count >= 2500
        observation = self._next_observation()

        # if new_location == goal:
        #     print('goal')

        return observation, reward, done, {}

    # def _reset(self):
    def reset(self):
        self.state.reset_state()
        self.round_count = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        return self.visualizer.visualize_combined_state(state=self.state).astype(np.uint8)

    def get_config_file(self):
        raise NotImplementedError("BasePrimalGymEnvironment is an abstract class")

    def _next_observation(self):

        # if True:
        #     return self.render()

        asset_location = self.state.assets[self.asset_number]

        ul_x = asset_location[0]
        ul_y = asset_location[1]
        # lr_x = asset_location[0] + 11
        # lr_y = asset_location[1] + 11
        lr_x = asset_location[0] + self.window_x
        lr_y = asset_location[1] + self.window_y

        observation = self.render()

        # paint the goal's location
        goal = self.state.goals[self.asset_number]
        observation[goal[0], goal[1], :] = [170, 0, 0]





        height, width, depth = observation.shape
        # new_observation = np.zeros((height + 10, width + 10, 3), dtype=np.uint8)
        new_observation = np.zeros((height + self.window_y - 1, width + self.window_x - 1, 3), dtype=np.uint8)
        offset_y, offset_x = int((self.window_y - 1) / 2), int((self.window_x - 1) / 2)
        # new_observation[5:height + 5, 5:width + 5, :] = observation
        new_observation[offset_y:height + offset_y, offset_x:width + offset_x, :] = observation



        # dir_x, dir_y = self.get_direction_point(asset_location, goal, (asset_location[0] - 5, asset_location[1] - 5, asset_location[0] + 5, asset_location[1] + 5))
        dir_x, dir_y = self.get_direction_point(asset_location, goal, (asset_location[0] - offset_y, asset_location[1] - offset_x, asset_location[0] + offset_y, asset_location[1] + offset_x))

        # paint the direction of our goal
        color = [209, 100, 233]
        # color = [channel - 40 for channel in color]
        # new_observation[dir_x + 5, dir_y + 5, :] = color  # cyan
        new_observation[dir_x + offset_x, dir_y + offset_y, :] = color  # cyan





        final_observation = new_observation[ul_x:lr_x, ul_y:lr_y]

        # paint our location in the center
        # final_observation[5, 5, :] = [39, 206, 216]  # sky blue
        final_observation[offset_x, offset_y, :] = [39, 206, 216]  # sky blue


        goal = self.state.goals[self.asset_number]

        # if abs(goal[0] - asset_location[0]) <= 5 and abs(goal[1] - asset_location[1]) <= 5:
        #     relative_goal_x = asset_location
        # dir_x, dir_y = self.get_direction_point(asset_location, goal, (asset_location[0]-5, asset_location[1]-5, asset_location[0]+5, asset_location[1]+5))
        #
        # # paint the direction of our goal
        # final_observation[dir_x, dir_y, :] = [209, 22, 233]  # cyan

        # from matplotlib import pyplot as plt
        # plt.imshow(observation, interpolation='nearest')
        # plt.show()
        # plt.imshow(final_observation, interpolation='nearest')
        # plt.show()

        asset_location = self.state.get_asset_location(self.asset_number)
        goal = self.state.get_asset_goal_location(self.asset_number)
        location_info = [i for i in itertools.chain(*[goal, asset_location])]

        answer = {
            'image': final_observation,
            'location': np.array([location_info])
        }

        # return final_observation

        return answer

    def get_intersection(self, pt1, pt2, pt3, pt4):

        # A = y2 - y1
        a1 = pt2[1] - pt1[1]
        # B = x1 - x2
        b1 = pt1[0] - pt2[0]
        # C = Ax1 + By1
        c1 = a1*pt1[0] + b1*pt1[1]

        # A = y2 - y1
        a2 = pt4[1] - pt3[1]
        # B = x1 - x2
        b2 = pt3[0] - pt4[0]
        # C = Ax1 + By1
        c2 = a2*pt3[0] + b2*pt3[1]

        # det = A1 * B2 - A2 * B1
        det = a1 * b2 - a2 * b1
        if det == 0:
            # Lines are parallel, I think I've guarded against this earlier
            print('ummm')
        else:
            x = (b2 * c1 - b1 * c2) / det
            y = (a1 * c2 - a2 * c1) / det

            return x, y


    def get_direction_point(self, asset_location, goal, rectangle):

        length = 100

        a_x = goal[0]
        a_y = goal[1]
        b_x = asset_location[0]
        b_y = asset_location[1]

        lenAB = sqrt(pow(a_x - b_x, 2.0) + pow(a_y - b_y, 2.0))

        if lenAB == 0.0:
            return asset_location[0], asset_location[1]

        x2 = b_x - (b_x - a_x) / lenAB * length;
        y2 = b_y - (b_y - a_y) / lenAB * length;

        min_x, min_y,  max_x,  max_y = rectangle
        x1, y1 = asset_location

        # calculate the slope
        # print(goal)
        if x2 - x1 != 0 and y2 - y1 != 0:
            # m = (y2 - y1) / (x2 - x1)
            #
            # if y2 < y1:
            #     y = m * (min_x - x1) + y1
            # else:
            #     y = m * (max_x - x1) + y1
            #
            # if x2 < x1:
            #     x = (min_y - y1) / m + x1
            # else:
            #     x = (max_y - y1) / m + x1
            #
            # print(f'* {m}, {x}, {y}')

            if x2 <= max_x and x2 >= min_x:
                if y2 > y1:
                    x, y = self.get_intersection((x1, y1), (x2, y2), (min_x, min_y), (max_x, min_y))
                else:
                    x, y = self.get_intersection((x1, y1), (x2, y2), (min_x, max_y), (max_x, max_y))
            elif y2 <= max_y and y2 >= min_y:
                if x2 > x1:
                    x, y = self.get_intersection((x1, y1), (x2, y2), (max_x, min_y), (max_x, max_y))
                else:
                    x, y = self.get_intersection((x1, y1), (x2, y2), (min_x, min_y), (min_x, max_y))
            else:

                # are in one of the corners
                if x2 < min_x and y2 < min_y:
                    # top left quad
                    tx1, ty1 = self.get_intersection((x1, y1), (x2, y2), (min_x, min_y), (max_x, min_y))
                    tx2, ty2 = self.get_intersection((x1, y1), (x2, y2), (min_x, min_y), (min_x, max_y))
                elif x2 > max_x and y2 < min_y:
                    # top right quad
                    tx1, ty1 = self.get_intersection((x1, y1), (x2, y2), (max_x, min_y), (max_x,max_y))
                    tx2, ty2 = self.get_intersection((x1, y1), (x2, y2), (min_x, min_y), (max_x, min_y))
                elif x2 < min_x and y2 > max_y:
                    # lower left quad
                    tx1, ty1 = self.get_intersection((x1, y1), (x2, y2), (min_x, min_y), (min_x, max_y))
                    tx2, ty2 = self.get_intersection((x1, y1), (x2, y2), (min_x, max_y), (max_x, max_y))
                else:
                    # lower right quad
                    tx1, ty1 = self.get_intersection((x1, y1), (x2, y2), (min_x, max_y), (max_x, max_y))
                    tx2, ty2 = self.get_intersection((x1, y1), (x2, y2), (max_x, min_y), (max_x, max_y))

                if abs(round(tx1)) > max_x or abs(round(ty1)) > max_y:
                    x = tx2
                    y = ty2
                else:
                    x = tx1
                    y = ty1

            # print(f'* {x}, {y}')

        elif x2 - x1 == 0:
            if y2 < y1:
                x = asset_location[0]
                y = asset_location[1] - 5
            else:
                x = asset_location[0]
                y = asset_location[1] + 5

            # print(f'* {x}, {y}')

        else:
            if x2 < x1:
                x = asset_location[0] - 5
                y = asset_location[1]
            else:
                x = asset_location[0] + 5
                y = asset_location[1]

            # print(f'* {x}, {y}')

        return round(x), round(y)


class EasyPrimalEnv(BasePrimalGymEnvironment):

    def __init__(self):
        BasePrimalGymEnvironment.__init__(self)

    def get_config_file(self):
        return "./configs/easy-training.yml"
        # return "./configs/training.yml"


class SingleAgentPrimalEnv(BasePrimalGymEnvironment):

    def __init__(self):
        BasePrimalGymEnvironment.__init__(self)

    def get_config_file(self):
        return "./configs/single-agent-training.yml"


class FourEnemyFourAllyAgentPrimalEnv(BasePrimalGymEnvironment):

    def __init__(self):
        BasePrimalGymEnvironment.__init__(self)

    def get_config_file(self):
        return "./configs/4_enemy_4_ally-training.yml"

class FourEnemyFourAllyFiftyWalllsAgentPrimalEnv(BasePrimalGymEnvironment):

    def __init__(self):
        BasePrimalGymEnvironment.__init__(self)

    def get_config_file(self):
        return "./configs/4e4a50w-training.yml"


if __name__ == '__main__':

    from PIL import Image
    import PIL

    # env = EasyPrimalEnv()
    env = SingleAgentPrimalEnv()
    # env = suite_gym.wrap_env(gym_env)

    print('Action Spec:')
    print(env.action_spec())

    print('Observation Spec:')
    print(env.observation_spec())

    # print((0, 10), env.get_direction_point((0, 0), (0, 10), (-5, -5, 5, 5)))
    # print((10, 0), env.get_direction_point((0, 0), (10, 0), (-5, -5, 5, 5)))
    # print((0, -10), env.get_direction_point((0, 0), (0, -10), (-5, -5, 5, 5)))
    # print((-10, 0), env.get_direction_point((0, 0), (-10, 0), (-5, -5, 5, 5)))
    # print((10, 10), env.get_direction_point((0, 0), (10, 10), (-5, -5, 5, 5)))
    # print((-10, -10), env.get_direction_point((0, 0), (-10, -10), (-5, -5, 5, 5)))
    # print((-10, -20), env.get_direction_point((0, 0), (-10, -20), (-5, -5, 5, 5)))
    # print((10, -20), env.get_direction_point((0, 0), (10, -20), (-5, -5, 5, 5)))
    # print((-10, 20), env.get_direction_point((0, 0), (-10, 20), (-5, -5, 5, 5)))
    # print((10, 20), env.get_direction_point((0, 0), (10, 20), (-5, -5, 5, 5)))
    # print((9, 9), env.get_direction_point((1, 10), (9, 9), (-4, 5, 6, 15)))
    # print('here')

    count = 0
    while True:
        observation, reward, done, _ = env.step(1)

        PIL.Image.fromarray(env.render())

        env_img = env.render()
        im = Image.fromarray(env_img.astype(np.uint8))
        im.save('env.png')
        im = Image.fromarray(observation.astype(np.uint8))
        im.save('observation.png')

        count += 1
        if count % 1000 == 0:
            print(count)

        env.reset()
