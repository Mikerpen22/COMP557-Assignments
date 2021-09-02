# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    
    """
    ### HELPERS
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # print(problem.__dict__)

    startState = problem.getStartState()    # Returns a tuple for the starting position
    visitedNodes =[]
    frontier = util.Stack()
    frontier.push([startState,[]])          # frontier[i] <- [(x,y), action_path taken so far till this node]

    while not frontier.isEmpty():
        # print(problem._visitedlist)
        curr_state, action_path = frontier.pop()
        
        # After popping, if not yet visited, add to visitedNodes
        if curr_state not in visitedNodes:
            visitedNodes.append(curr_state)

            if problem.isGoalState(curr_state):
                # print(action_path)
                return action_path
            else:
                for nbr_info in problem.getSuccessors(curr_state):      
                    ### GetSeccessors returns (nextState, direction to nextState, cost)
                    frontier.push([nbr_info[0], action_path+[nbr_info[1]]])
    
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    #to be explored (FIFO)
    frontier = util.Queue()
    
    #previously expanded states (for cycle checking), holds states
    exploredNodes = []
    
    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)
    
    frontier.push(startNode)
    
    while not frontier.isEmpty():
        #begin exploring first (earliest-pushed) node on frontier
        currentState, actions, currentCost = frontier.pop()
        
        if currentState not in exploredNodes:
            #put popped node state into explored list
            exploredNodes.append(currentState)
            # print(currentState, actions, currentCost)
            if problem.isGoalState(currentState):
                return actions
            else:
                #list of (successor, action, stepCost)
                successors = problem.getSuccessors(currentState)
                
                for succState, succAction, succCost in successors:
                    newAction = actions + [succAction]
                    newCost = currentCost + succCost
                    newNode = (succState, newAction, newCost)

                    frontier.push(newNode)

    return actions

    "*** YOUR CODE HERE ***" 
    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    startState = problem.getStartState()    # Returns a tuple for the starting position
    visitedNodes = {}
    frontier = util.PriorityQueue()
    frontier.push([startState, [], 0], 0)   # frontier.push(item, priority)
                                            # item: [state, actions, cost] 

    while not frontier.isEmpty():
        # print(problem._visitedlist)
        curr_state, action_path, curr_cost = frontier.pop()

        if(curr_state not in visitedNodes.keys() or curr_cost < visitedNodes[curr_state]):
            visitedNodes[curr_state] = curr_cost

            if problem.isGoalState(curr_state):
                # print(action_path)    
                # print(visitedNodes)            
                return action_path
            else:
                for nbr_info in problem.getSuccessors(curr_state):      
                    ### GetSeccessors returns (nextState, direction to nextState, cost)
                    frontier.update([nbr_info[0], action_path+[nbr_info[1]], curr_cost+nbr_info[2]], curr_cost+nbr_info[2])
    return action_path
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined() 

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    startState = problem.getStartState()    # Returns a tuple for the starting position
    visitedNodes = {}                       # {current state: current cost}
    frontier = util.PriorityQueue()
    frontier.push([startState, [], 0], 0)   # frontier.push(item, priority)
                                            # item: [state, actions, cost] 

    while not frontier.isEmpty():
        # print(problem._visitedlist)
        curr_state, action_path, curr_cost = frontier.pop()    

        cornor_problem = False
        if len(curr_state) >= 2 and type(curr_state[1]) == list: # to check if heuristic to be used is cornor problem.
            cornor_problem = True

        if cornor_problem:
            visitedNodes[curr_state[0]] = curr_cost
        else:
            visitedNodes[curr_state] = curr_cost

        print(curr_state)
        
        if problem.isGoalState(curr_state):        
            return action_path
        else:
            for nbr_info in problem.getSuccessors(curr_state):      
                ## GetSeccessors returns (nextState, direction to nextState, cost)
                # print(nbr_info)
                key = nbr_info
                if cornor_problem:
                    key = nbr_info[0][0]
                else:
                    key = nbr_info[0]

                if (key not in visitedNodes.keys() or 
                        visitedNodes[key] > curr_cost+nbr_info[2]+heuristic(key, problem)-heuristic(curr_state, problem)):
                        heuristic_diff = heuristic(key, problem) - heuristic(curr_state, problem)
                        frontier.update([nbr_info[0], action_path+[nbr_info[1]], curr_cost+nbr_info[2]+heuristic_diff], 
                                                                                curr_cost+nbr_info[2]+heuristic_diff)
                        visitedNodes[key] = curr_cost+nbr_info[2]+heuristic_diff
               
                

    return action_path
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
