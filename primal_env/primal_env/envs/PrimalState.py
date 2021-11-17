import numpy as np
from random import randrange
from math import sqrt
from primal_env.envs.PathingExpert import a_star

class PrimalState:

    def __init__(self, config):

        # size information
        self.width = config['simulator']['width']
        self.height = config['simulator']['height']

        # sim information
        self.num_enemies = config['simulator']['num-enemies']
        self.num_allies = config['simulator']['num-allies']
        self.num_assets = config['simulator']['num-assets']
        self.num_barriers = config['simulator']['num-barriers']

        # location information
        self.enemy_locations = np.zeros((self.height, self.width))
        self.ally_locations = np.zeros((self.height, self.width))
        self.asset_locations = np.zeros((self.height, self.width))
        self.barrier_locations = np.zeros((self.height, self.width))

        # influence information
        self.enemy_influence_distance = config['simulator']['enemy-influence-distance']
        self.ally_influence_distance = config['simulator']['ally-influence-distance']
        self.enemy_influence = np.zeros((self.height, self.width))
        self.ally_influence = np.zeros((self.height, self.width))
        self.mixed_influence = np.zeros((self.height, self.width))

        # cost information
        costs = config['simulator']['costs']
        self.cost_normal_movement = costs['normal_movement']
        self.cost_ally_influence = costs['ally_influence']
        self.cost_enemy_influence = costs['enemy_influence']
        self.cost_mixed_influence = costs['mixed_influence']
        self.cost_barrier = costs['barrier']
        self.cost_asset = costs['asset']
        self.cost_ally = costs['ally']
        self.cost_enemy = costs['enemy']

        self.goal_locations = np.zeros((self.height, self.width))
        self.goals = []
        self.assets = []

        self.cost_matrix = np.zeros((self.height, self.width))

        # set the mappings of state names and state values
        self.state_mapping = {}
        state_mapping_config = config['state']['mappings']
        for key in state_mapping_config:
            value = state_mapping_config[key]
            self.state_mapping[key] = value
            self.state_mapping[value] = key

        self.reset_state()

    def get_combined_state(self):
        answer = np.zeros((self.height, self.width))

        answer[self.enemy_influence == 1] = self.state_mapping['enemy_influence']
        answer[self.ally_influence == 1] = self.state_mapping['ally_influence']
        answer[self.mixed_influence == 1] = self.state_mapping['mixed_influence']

        answer[self.enemy_locations == 1] = self.state_mapping['enemy']
        answer[self.ally_locations == 1] = self.state_mapping['ally']

        answer[self.barrier_locations == 1] = self.state_mapping['barrier']
        answer[self.asset_locations == 1] = self.state_mapping['asset']

        # remember, goals are numbered by the asset they belong to.
        # we also start at 0 for assets, all others are -1 if not goal locations
        answer[self.goal_locations >= 0] = self.state_mapping['goal']

        return answer

    def reset_state(self):

        self.__reset_info(self.barrier_locations, self.num_barriers)
        self.__reset_info(self.ally_locations, self.num_allies, self.ally_influence, self.ally_influence_distance)
        self.__reset_info(self.enemy_locations, self.num_enemies, self.enemy_influence, self.enemy_influence_distance)

        # don't forget to get the mixed influence
        self.mixed_influence.fill(0)
        self.mixed_influence[(self.enemy_influence == 1) & (self.ally_influence == 1)] = 1

        # need to calculate cost matrix before assets because assets uses it
        self.cost_matrix = self.__reset_cost_matrix()

        self.__reset_asset_info()

    def __reset_info(self, field, count, influence=None, influence_distance=0):
        # zero out the field
        field.fill(0)
        if influence is not None:
            influence.fill(0)

        # keep trying to add until we have the correct amount
        while field.sum() < count:
            y, x = randrange(self.height - 1), randrange(self.width - 1)
            if self.is_empty_location(y, x):
                field[y][x] = 1

                # also set the influence if needed
                if influence is not None:
                    # TODO: ew, need to vectorize this
                    for i in range(self.height):
                        for j in range(self.width):
                            if sqrt((i - y)**2 + (j - x)**2) < influence_distance:
                                influence[i][j] = 1

    def __reset_asset_info(self):
        self.asset_locations.fill(0)
        # we are filling goal locations with -1 because the asset numbers start at 0
        self.goal_locations.fill(-1)
        self.goals = []
        self.assets = []

        asset_num = 0

        # create all of the allies and their corresponding goals
        while np.count_nonzero(self.asset_locations) < self.num_assets:
            y, x = randrange(self.height - 1), randrange(self.width - 1)
            if self.is_empty_location(y, x):
                self.asset_locations[y][x] = 1
                self.assets.append((y, x))
                # self.assets.append(BaseAgent(y, x))

                goal_placed = False
                while not goal_placed:
                    goal_y, goal_x = randrange(self.height - 1), randrange(self.width - 1)
                    # path = a_star((y, x), (goal_y, goal_x), self.cost_matrix)
                    # we need to check if it is physically empty as well as if anyone else's goal is here
                    if self.is_empty_location(goal_y, goal_x) and self.goal_locations[goal_y][goal_x] == -1:
                        # TODO: need to check to see if there is actually a path from the agent to their goal
                        # set the goal location to the number of the ally and save the tuple
                        self.goal_locations[goal_y][goal_x] = asset_num
                        self.goals.append((goal_y, goal_x))

                        # update record keeping
                        asset_num += 1
                        goal_placed = True

    def move_asset(self, asset_number, location):
        prev_y, prev_x = self.assets[asset_number]
        new_y, new_x = location

        self.asset_locations[prev_y][prev_x] = 0
        self.asset_locations[new_y][new_x] = 1
        self.assets[asset_number] = (new_y, new_x)

        return prev_y, prev_x

    def get_cost_matrix(self):
        # copy the static bits
        matrix = self.cost_matrix.copy()

        # we need to insert this information in every time because the assets are always moving
        matrix[self.asset_locations == 1] = self.cost_asset

        return matrix

    def is_empty_location(self, y, x):

        # lets pretend there is a wall around the field
        if x < 0 or y < 0 or y >= len(self.enemy_locations) or x >= len(self.enemy_locations[0]):
            return False

        # we don't look at goals here because they don't take up space
        return self.enemy_locations[y][x] == 0 and \
               self.ally_locations[y][x] == 0 and \
               self.asset_locations[y][x] == 0 and \
               self.barrier_locations[y][x] == 0

    def __reset_cost_matrix(self):
        cost = np.full(self.barrier_locations.shape, self.cost_normal_movement)
        cost[(self.ally_influence == 1) & (self.enemy_influence == 1)] = self.cost_mixed_influence
        cost[self.ally_influence == 1] = self.cost_ally_influence
        cost[self.enemy_influence == 1] = self.cost_enemy_influence
        cost[self.barrier_locations == 1] = self.cost_barrier
        cost[self.enemy_locations == 1] = self.cost_enemy
        cost[self.ally_locations == 1] = self.cost_ally

        return cost

    def get_asset_goal_location(self, asset_number):
        # goal_location = np.where(self.goal_locations == asset_number)
        # goal = (goal_location[0][0], goal_location[1][0])

        goal = self.goals[asset_number]

        return goal

    def get_asset_location(self, asset_number):
        asset_location = self.assets[asset_number]

        return asset_location

    def location_inside_area(self, location):
        location_y, location_x = location

        return 0 <= location_x < self.width and 0 <= location_y < self.height

    def get_asset_distance_from_goal(self, asset_number):
        goal = self.get_asset_goal_location(asset_number)
        location = self.get_asset_location(asset_number)
        distance = sqrt((goal[0] - location[0]) ** 2 + (goal[1] - location[1]) ** 2)

        return distance
