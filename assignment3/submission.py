import collections
import util
import math
import random

############################################################
# Problem 4.1.1


def computeQ(mdp, V, state, action):
    """
    Return Q(state, action) based on V(state).  Use the properties of the
    provided MDP to access the discount, transition probabilities, etc.
    In particular, MDP.succAndProbReward() will be useful (see util.py for
    documentation).  Note that |V| is a dictionary.  
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    Q_sum = 0
    for newState, prob, reward in mdp.succAndProbReward(state, action):
        Q_sum += prob * (reward + mdp.discount() * V[newState])

    return Q_sum
    # END_YOUR_CODE

############################################################
# Problem 4.1.2


def policyEvaluation(mdp, V, pi, epsilon=0.001):
    """
    Return the value of the policy |pi| up to error tolerance |epsilon|.
    Initialize the computation with |V|.  Note that |V| and |pi| are
    dictionaries.
    """
    # BEGIN_YOUR_CODE (around 7 lines of code expected)
    mdp.computeStates()
    states = mdp.states
    for state in states:
        V[state] = 0

    err = 100
    while err >= epsilon:
        newValues = {}
        for state in states:
            newValues[state] = computeQ(mdp, V, state, pi[state])
            err = abs(newValues[state] - V[state])

        # Exit the loop if err is within epsilon
        if(err < epsilon):
            return newValues
        # If not, update V and continue the loop
        V = newValues
    return V

    # END_YOUR_CODE

############################################################
# Problem 4.1.3


def computeOptimalPolicy(mdp, V):
    """
    Return the optimal policy based on V(state).
    You might find it handy to call computeQ().  Note that |V| is a
    dictionary.
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    # For each state, go through all its actions and computeQ to find the action based on V from policy evaluation
    pi_optimal = {}
    for state in V:
        Q_max = float("-inf")
        Q_max_action = None
        for action in mdp.actions(state):
            Q = computeQ(mdp, V, state, action)
            if(Q > Q_max):
                Q_max = Q
                Q_max_action = action

        # Update policy pi to the best action we find
        pi_optimal[state] = Q_max_action

    return pi_optimal
    # END_YOUR_CODE

############################################################
# Problem 4.1.4


class PolicyIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        states = mdp.states
        # compute |V| and |pi|, which should both be dicts

        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        # Initialize states for first iteration
        V, pi = {}, {}
        for state in states:
            V[state] = 0
            pi[state] = None
        pi_updated = {}

        # Continue policy iteration till policy converges
        while pi != pi_updated:
            pi = pi_updated
            pi_updated = computeOptimalPolicy(mdp, V)
            V = policyEvaluation(mdp, V, pi_updated, epsilon)

        # END_YOUR_CODE
        self.pi = pi
        self.V = V

############################################################
# Problem 4.1.5


class ValueIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        states = mdp.states

        # BEGIN_YOUR_CODE (around 10 lines of code expected)
        V, pi = {}, {}
        # Initialize Values & Policies
        for state in states:
            V[state] = 0
            pi[state] = None

        err = float("inf")
        while err > epsilon:
            # Generate policy based on V state
            # For each state, pick an action and see if value can be improved

            # The diff between Value Iteration & Policy Iteration is mainly on the next line
            # We take max over all possible actions in Value Iteration
            pi = computeOptimalPolicy(mdp, V)
            V_updated = {}
            for state in V.keys():
                V_updated[state] = computeQ(mdp, V, state, pi[state])
                err = abs(V_updated[state] - V[state])
            V = V_updated

        # END_YOUR_CODE
        self.pi = pi
        self.V = V

############################################################
# Problem 4.1.6

# If you decide 1f is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 1f is false, construct a
# counterexample by filling out this class and returning an alpha value in
# counterexampleAlpha().


class CounterexampleMDP(util.MDP):
class CounterexampleMDP(util.MDP):
    def __init__(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return
        # END_YOUR_CODE

    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 0
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return ['a']
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        if state == 0:
            return [(1, 0.2, 10), (2, 0.8, 3)]

        # End state => directly return
        return []
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        # No discount factor in this example
        return 1
        # END_YOUR_CODE


def counterexampleAlpha():
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return 0.2
    # END_YOUR_CODE

############################################################
# Problem 4.2.1


class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        # total, next card (if any), multiplicity for each card
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 50 lines of code expected)

        result = []
        if action == 'Quit':
            if state[2] and sum(state[2]) == 0:
                return []
            elif not state[2]:
                return []
            result.append(((state[0], None, None), 1.0, state[0]))

        elif state[2] == (0,) * len(self.cardValues):
            result.append(((state[0], None, None), 1, state[0]))

        elif action == 'Take':
            # check if peeked or not
            # if peeked, you will have to draw that card.
            # else we can take the card or choose to peek

            # the next card is not peeked
            # (1,0,1,1) =>
            if state[2] is not None and sum(state[2]) > 0:
                # cardDeck = list(state[2])   # cast type to list in order to modified it
                if state[1] is None:
                    for i in range(len(state[2])):
                        cardDeck = list(state[2])   # cast type to list in order to modified it
                        total_sum = sum(state[2])
                        if cardDeck[i] != 0:
                            prob = cardDeck[i] / total_sum
                            cardDeck[i] -= 1
                            # Check if busted or not after taking the card
                            if sum(cardDeck) != 0:
                                if state[0] + self.cardValues[i] < self.threshold:
                                    cardDecks = tuple(cardDeck)
                                    result.append(
                                        ((state[0]+self.cardValues[i], None, cardDecks), prob, 0))
                                elif state[0] + self.cardValues[i] == self.threshold:
                                    cardDecks = tuple(cardDeck)
                                    # print("exit 1")
                                    result.append(
                                        ((state[0]+self.cardValues[i], None, None), prob, state[0]+self.cardValues[i]))
                                else:
                                    # print("busted")
                                    result.append(((state[0]+self.cardValues[i], None, None), prob, 0))
                            elif sum(cardDeck) == 0:
                                result.append(
                                        ((state[0]+self.cardValues[i], None, None), 1, state[0]+self.cardValues[i]))
                                
                else:
                    cardDeck = list(state[2])   # cast type to list in order to modified it
                    peekingIndex = state[1]
                    cardDeck[peekingIndex] -= 1

                    if state[0] + self.cardValues[peekingIndex] < self.threshold:
                        cardDecks = tuple(cardDeck)
                        print("pick peeked card and not busted")
                        result.append(
                            ((state[0] + self.cardValues[peekingIndex], None, cardDecks), 1, 0))
                    else:
                        print("pick peeked card and busted")
                        result.append(((0, None, None), 1.0, 0))

        elif action == 'Peek':
            # accomodate peeking cost
            if state[2] is not None:
                cardDeck = list(state[2])
                for i in range(len(cardDeck)):
                    total_sum = sum(cardDeck)
                    if cardDeck[i] != 0:
                        prob = cardDeck[i]/total_sum
                        cardDecks = tuple(cardDeck)
                        result.append(
                            ((state[0], i, cardDecks), prob, state[0] - self.peekCost))

        return result

        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 4.2.2


def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    return BlackjackMDP([1,2,5],3,10,1)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE
