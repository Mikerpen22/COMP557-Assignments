# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]  # Evaluate each action's score
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        ## Be careful not to get too close to ghosts
        avgDistToGhost = 0
        distToGhosts = []
        for ghostState in newGhostStates:
            distToGhosts.append(manhattanDistance(newPos, ghostState.getPosition()))
        minDistToGhosts = min(distToGhosts)
        avgDistToGhost = sum(distToGhosts) / len(distToGhosts)

        ## Make sure we get closer to the closest food
        distToFoods, distToNewFoods = [], []
        for food in list(currentGameState.getFood()):
            distToFoods.append(manhattanDistance(currentGameState.getPacmanPosition(), food))
        for food in list(newFood):
            distToNewFoods.append(manhattanDistance(newPos, food))
        minDistToFood = min(distToFoods)
        minDistToNewFood = min(distToNewFoods)

        scoreDiff = successorGameState.getScore() - currentGameState.getScore()

        direction = currentGameState.getPacmanState().getDirection()

        if minDistToGhosts <= 1 or action == Directions.STOP:
            return 0
        elif avgDistToGhost >= 2:
            return 2
        elif scoreDiff > 0:
            return 8
        elif minDistToNewFood - minDistToFood < 0:
            return 8
        elif action == direction:
            return 6
        else:
            return 1

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        ## agent_idx: indicating whether its Pac's turn or ghost's turn
        def minmax(state, depth, agent_idx):
            ## Directly evaluate the state if either the we reach the depth limit or leaves state
            if (state.isWin() or state.isLose() or depth == 0):
                return self.evaluationFunction(state)

            ## At max agent's state(PAC)
            elif agent_idx == 0:
                legalActions = state.getLegalActions(agent_idx)
                ## recursively call minmax on the ghosts:
                ## pass in 1 for the agent_idx so we can switch to the else loop dealing with ghosts
                return max(minmax(state.generateSuccessor(agent_idx, action), depth, 1) for action in legalActions)

                ## At min agent's(ghost) state
            else:
                if agent_idx < state.getNumAgents() - 1:
                    legalActions = state.getLegalActions(agent_idx)
                    return min(minmax(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1) for action in
                               legalActions)
                ## Next move will be Pac's, so we...
                ## 1. Decrease the depth by 1
                ## 2. Pass in 0 for agent_idx to indicate its Pac's turn
                else:
                    legalActions = state.getLegalActions(agent_idx)
                    return min(
                        minmax(state.generateSuccessor(agent_idx, action), depth - 1, 0) for action in legalActions)

                    ## First execution from Pacman

        best_action = Directions.EAST  # This is just a initial value, can be EAST, SOUTH, WEST, NORTH or even STOP
        legalActions = gameState.getLegalActions(0)  # At the root(PAC) now, get its legal actions
        max_action_val = float("-inf")
        ## Find the max value through minmax for Pac to choose
        for action in legalActions:
            temp_max_action_val = minmax(gameState.generateSuccessor(0, action), self.depth,
                                         1)  # Next move will be the first ghost, pass 1 for agent_idx to show that
            if (temp_max_action_val > max_action_val):
                max_action_val = temp_max_action_val
                best_action = action
        return best_action
        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        ## agent_idx: indicating whether its Pac's turn or ghost's turn
        def ABPruneMinmax(state, depth, agent_idx, a, b):
            ## Directly evaluate the state if either the we reach the depth limit or leaves state
            if (state.isWin() or state.isLose() or depth == 0):
                return self.evaluationFunction(state)

            ## At max agent's state(PAC)
            elif agent_idx == 0:
                legalActions = state.getLegalActions(agent_idx)
                ## recursively call minmax on the ghosts:
                ## pass in 1 for the agent_idx so we can switch to the else loop dealing with ghosts
                max_value = float("-inf")
                for action in legalActions:
                    max_value = max(max_value,
                                    ABPruneMinmax(state.generateSuccessor(agent_idx, action), depth, 1, a, b))
                    if max_value > b:  # max_value is over the upper limit
                        return max_value
                    a = max(a, max_value)
                return max_value

            ## At min agent's(ghost) state
            else:
                min_value = float("inf")
                if agent_idx < state.getNumAgents() - 1:
                    legalActions = state.getLegalActions(agent_idx)
                    for action in legalActions:
                        min_value = min(min_value,
                                        ABPruneMinmax(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1,
                                                      a, b))
                        ## If min_value is even less than my lower limit a, return it directly and forget about the rest
                        if min_value < a:
                            return min_value
                        ## Updating the upper limit b
                        b = min(b, min_value)
                    return min_value

                ## Next move will be Pac's, so we...
                ## 1. Decrease the depth by 1
                ## 2. Pass in 0 for agent_idx to indicate its Pac's turn
                else:
                    legalActions = state.getLegalActions(agent_idx)
                    # return min(minmax(state.generateSuccessor(agent_idx, action), depth-1, 0) for action in legalActions) 
                    for action in legalActions:
                        min_value = min(min_value,
                                        ABPruneMinmax(state.generateSuccessor(agent_idx, action), depth - 1, 0, a, b))
                        if min_value < a:
                            return min_value
                        b = min(b, min_value)
                    return min_value

        ## First execution at Pacman
        best_action = Directions.EAST  # This is just a initial value, can be EAST, SOUTH, WEST, NORTH or even STOP

        a = float("-inf")
        b = float("inf")
        best_action_val = float("-inf")
        legalActions = gameState.getLegalActions(0)  # At the root(PAC) now, get its legal actions

        ## Find the max value through minmax for Pac to choose
        for action in legalActions:
            temp_max_action_val = ABPruneMinmax(gameState.generateSuccessor(0, action), self.depth, 1, a,
                                                b)  # Next move will be the first ghost, pass 1 for agent_idx to show that
            if (temp_max_action_val > best_action_val):
                best_action_val = temp_max_action_val
                best_action = action
            if best_action_val > b:
                return best_action_val
            a = max(a, best_action_val)
        return best_action
        # util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        ## agent_idx: indicating whether its Pac's turn or ghost's turn
        def expectiMax(state, depth, agent_idx):
            ## Directly evaluate the state if either the we reach the depth limit or leaves state
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            ## At max agent's state(PAC)
            elif agent_idx == 0:
                legal_actions = state.getLegalActions(agent_idx)
                ## recursively call minmax on the ghosts:
                ## pass in 1 for the agent_idx so we can switch to the else loop dealing with ghosts
                max_value = float("-inf")
                for action in legal_actions:
                    max_value = max(max_value, expectiMax(state.generateSuccessor(agent_idx, action), depth, 1))
                return max_value

            ## At min agent's(ghost) state
            else:
                expectedValue = 0
                min_value = float("inf")
                if agent_idx < state.getNumAgents() - 1:
                    legal_actions = state.getLegalActions(agent_idx)
                    succ_state = [state.generateSuccessor(agent_idx, x) for x in legal_actions]
                    for action in legal_actions:
                        expectedValue += expectiMax(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1)
                    return expectedValue / len(succ_state)
                ## Next move will be Pac's, so we...
                ## 1. Decrease the depth by 1
                ## 2. Pass in 0 for agent_idx to indicate its Pac's turn
                else:
                    legal_actions = state.getLegalActions(agent_idx)
                    succ_state = [state.generateSuccessor(agent_idx, x) for x in legal_actions]
                    # return min(minmax(state.generateSuccessor(agent_idx, action), depth-1, 0) for action in legalActions)
                    for action in legal_actions:
                        expectedValue += expectiMax(state.generateSuccessor(agent_idx, action), depth - 1, 0)
                    return expectedValue / len(succ_state)


        ## First execution at Pacman
        best_action = Directions.EAST  # This is just a initial value, can be EAST, SOUTH, WEST, NORTH or even STOP
        best_action_val = float("-inf")
        legalActions = gameState.getLegalActions(0)  # At the root(PAC) now, get its legal actions

        ## Find the max value through minmax for Pac to choose
        for action in legalActions:
            temp_max_action_val = expectiMax(gameState.generateSuccessor(0, action), self.depth, 1)
            # Next move will be the first ghost, pass 1 for agent_idx to show that
            if temp_max_action_val > best_action_val:
                best_action_val = temp_max_action_val
                best_action = action
        return best_action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
