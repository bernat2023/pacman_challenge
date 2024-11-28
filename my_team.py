# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util
from layout import Layout
from capture_agents import CaptureAgent
from game import Directions, Actions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class OptimizeCaptureAgent(CaptureAgent):
    """
    Improved Reflex Agent with smarter behaviors for both offensive and defensive strategies.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.layout = Layout("", game_state.data.layout.layout_text)

    def choose_action(self, game_state):
        """
        Picks the best action based on enhanced evaluation.
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, action) for action in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # If low food left, return to start for safety
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor with position adjustments.
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Override in subclasses for specific strategies.
        """
        return util.Counter()

    def get_weights(self, game_state, action):
        """
        Override in subclasses for specific strategies.
        """
        return {}

class OffensiveAgent(OptimizeCaptureAgent):
    """
    Offensive Agent with safe-zone logic for defensive retreats and immediate re-entry to offense.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.food_collected = 0
        self.last_position = None
        self.last_action = None

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()

        # Track food
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        # Distance to nearest food
        if food_list:
            features['distance_to_food'] = min(self.get_maze_distance(my_pos, food) for food in food_list)

        # Ghost avoidance
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [enemy for enemy in enemies if not enemy.is_pacman and enemy.get_position() is not None]
        if ghosts:
            ghost_distances = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]
            features['ghost_distance'] = min(ghost_distances)

            # If near a ghost, prioritize escaping
            if features['ghost_distance'] < 3:
                features['avoid_ghost'] = 1

        # Penalize stopping
        if action == Directions.STOP:
            features['stop'] = 1

        # Penalize reversing direction (when not needed)
        reverse_direction = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse_direction:
            features['reverse'] = 1

        # Add defensive retreat feature
        if self.food_collected >= 3:
            if self.on_defensive_side(my_pos, game_state):
                # Minimal penalty for staying in defensive zone
                features['in_safe_zone'] = 1
            else:
                # Strong incentive to retreat to the defensive side
                features['retreat_to_safe_zone'] = -self.get_maze_distance(my_pos, self.start)

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,
            'distance_to_food': -1,
            'ghost_distance': 20,
            'avoid_ghost': 1000,
            'stop': -1000,
            'reverse': -5,
            'in_safe_zone': 100,                # Reward for being in the defensive zone
            'retreat_to_safe_zone': 50          # Incentivize retreating to the safe zone
        }

    def choose_action(self, game_state):
        """
        Implements safe-zone logic and efficient re-entry to the offensive side.
        """
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return Directions.STOP

        # Evaluate all actions
        values = [self.evaluate(game_state, action) for action in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Get current position
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Detect loop (oscillating back and forth)
        if self.last_position and my_pos == self.last_position:
            if len(best_actions) > 1:
                # Remove the last action from the best choices, if alternatives exist
                if self.last_action in best_actions:
                    best_actions.remove(self.last_action)
            else:
                # If no alternatives, allow reversing to prevent stopping
                best_actions = actions

        # Never choose STOP unless it's the only option
        if Directions.STOP in best_actions and len(best_actions) > 1:
            best_actions.remove(Directions.STOP)

        # Choose the best action from the refined list
        chosen_action = random.choice(best_actions)

        # Update position and action memory
        self.last_position = my_pos
        self.last_action = chosen_action

        # Update food count when food is consumed
        successor = self.get_successor(game_state, chosen_action)
        food_list = self.get_food(successor).as_list()
        if len(self.get_food(game_state).as_list()) > len(food_list):
            self.food_collected += 1

        # Reset food count when entering the defensive side
        if self.food_collected >= 3 and self.on_defensive_side(successor.get_agent_position(self.index), game_state):
            self.food_collected = 0

        return chosen_action

    def on_defensive_side(self, position, game_state):
        """
        Determines if the given position is on the defensive side.
        """
        midline = (game_state.data.layout.width // 2)
        return position[0] < midline

class DefensiveAgent(OptimizeCaptureAgent):
    """
    Improved Defensive Agent that efficiently blocks enemy Pacman.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Defensive mode
        features['on_defense'] = 1 if not my_state.is_pacman else 0

        # Track invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if invaders:
            dists = [self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]
            features['invader_distance'] = min(dists)

        # Capsule proximity (to prevent invaders from getting it)
        capsules = self.get_capsules_you_are_defending(successor)
        if capsules:
            features['distance_to_capsule'] = min(self.get_maze_distance(my_pos, capsule) for capsule in capsules)

        # Avoid stopping or reversing
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'distance_to_capsule': -2,
            'stop': -100,
            'reverse': -2
        }
