import numpy as np


class Visualizer:

    def __init__(self, config):

        self.scale = config['scale']

        self.colors = {}
        color_config = config['colors']
        # loop through all of the colors and create a numpy array for them
        for key in color_config:
            self.colors[key] = np.fromstring(color_config[key], dtype=int, sep=',')

        # we start out without actually visualizing anything so things are initialized to None
        self.state = None
        self.pixels = None
        self.screen = None
        self.visualization_initialized = False

    # def visualize(self, state=None, show=True):
    #     if show and state is not None:
    #         # now we can start importing pixel and pygame classes
    #         from pixel_class import Pixel
    #         from pygame.display import set_mode, flip
    #         from pygame import init as pygame_init
    #
    #         pygame_init()
    #
    #         self.flip = flip
    #         self.state = state
    #         self.screen = set_mode((state.width * self.scale, state.height * self.scale))
    #         self.screen.fill((0, 0, 0))
    #         combined_state = state.get_combined_state()
    #         self.pixels = [[Pixel(x, y, self.screen, self.colors[state.state_mapping[combined_state[y][x]]], self.scale)
    #                         for x in range(state.width)] for y in range(state.height)]
    #         self.visualization_initialized = True
    #
    # def move_asset(self, previous, next):
    #     # we haven't been told we need to visualize anything yet
    #     if not self.visualization_initialized:
    #         return
    #
    #     combined_state = self.state.get_combined_state()
    #     old_color = self.colors[self.state.state_mapping[combined_state[previous[0]][previous[1]]]]
    #     self.pixels[previous[0]][previous[1]].change_state(old_color)
    #     self.pixels[next[0]][next[1]].change_state(self.colors['asset'])
    #
    #     self.flip()

    def visualize_combined_state(self, state, file_name=None):
        # make sure it has been initialized, be we don't want to unless this is called
        from matplotlib import pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        combined_state = state.get_combined_state()
        data = np.full((state.height, state.width, 3), 255)

        for key in self.colors:
            locations = np.where(combined_state == state.state_mapping[key])
            data[locations[0], locations[1], :] = self.colors[key]

        # goals were getting overwritten, so do it again
        locations = np.where(state.goal_locations == 1)
        data[locations[0], locations[1], :] = self.colors['goal']

        return data


    # def reset(self):
    #     # we haven't been told we need to visualize anything yet
    #     if not self.visualization_initialized:
    #         return
    #
    #     from pixel_class import Pixel
    #     # from pygame.display import set_mode, flip
    #     # from pygame import init as pygame_init
    #
    #     # pygame_init()
    #
    #     self.screen.fill((0, 0, 0))
    #     combined_state = self.state.get_combined_state()
    #     self.pixels = [[Pixel(x, y, self.screen, self.colors[self.state.state_mapping[combined_state[y][x]]], self.scale)
    #                     for x in range(self.state.width)] for y in range(self.state.height)]
