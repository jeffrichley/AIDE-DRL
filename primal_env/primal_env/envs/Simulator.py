import numpy as np

class BasicSimulator():

    def __init__(self, trainer, state, width, height):

        self.trainer = trainer
        self.state = state

        # screen information
        # self.width = width
        # self.height = height

        # sim information
        # self.barrier_locations = state.barrier_locations
        # self.asset_locations = state.asset_locations
        # self.ally_locations = state.ally_locations
        # self.enemy_locations = state.enemy_locations
        # self.goal_locations = state.goal_locations

        # get the field ready to play
        self.state.reset_state()

    def move_asset(self, asset_number, new_location):
        # we need to make sure it is an empty location before we accept the move
        moved = True
        if self.valid_action(asset_number, new_location):
            self.state.move_asset(asset_number, new_location)
            reward = self.trainer.get_reward(asset_number, self.state)
        else:
            moved = False
            reward = self.trainer.get_reward_for_location(location=new_location, state=self.state, asset_number=asset_number)
            # do the bad move
            # previous_location = self.state.move_asset(asset_number, new_location)
            # reward = self.trainer.get_reward(asset_number, self.state)
            # roll back the bad move
            # self.state.move_asset(asset_number, previous_location)

        return moved, reward

    def valid_action(self, asset_number, new_location):

        y = new_location[0]
        x = new_location[1]

        asset_y, asset_x = self.state.get_asset_location(asset_number)

        return (y == asset_y and x == asset_x) or \
               (self.state.is_empty_location(y, x) and self.state.location_inside_area(new_location))


