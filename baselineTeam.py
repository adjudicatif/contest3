# baselineTeam.py
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Actions, Directions
import game
from util import nearestPoint


##########
# Support#
##########
#########
class AgentBelieves:
  def register(self, gameState, indices):
    if not hasattr(self, "Bs"):
      self.Bs = {index: AgentBelief(gameState, index) for index in indices}
                 
  def updateBelief(self, myPos, gameState):
    return [B.updateBelief(myPos, gameState) for B in self.Bs.values()]
            
  def getbestAgentsPositions(self, gameState):
    return [self.getAgentPosition(index, gameState) for index in self.Bs]
    
  def getAgentPosition(self, index, gameState):
    if gameState.getAgentPosition(index) == None:
      return self.Bs[index].B.argMax()
    else:
      return gameState.getAgentPosition(index)

class AgentBelief:
  def __init__(self, gameState, index):
    self.index = index
    self.grid = gameState.getWalls().asList(key=False)
    self.B = util.Counter()
    for pos in self.grid:
      self.B[pos] = 1.0 / len(self.grid)
      
  def updateBelief(self, myPos, gameState):    
    distanceFn = distanceCalculator.manhattanDistance
    agentPos = gameState.getAgentPosition(self.index)    
    #Advance in time
    B_prime = util.Counter()
    for prev_pos in self.grid:
      neighbors = Actions.getLegalNeighbors(prev_pos, gameState.getWalls())
      if len(neighbors) > 0: 
        p = 1.0 / len(neighbors)
        for pos in neighbors:
          B_prime[pos] += self.B[prev_pos] * p
    #Consider the observation
    B = util.Counter()
    for pos in B_prime:
      if agentPos == None:
        observation = gameState.getAgentDistances()[self.index]
        B[pos] = B_prime[pos] * gameState.getDistanceProb(distanceFn(myPos, pos), observation)
      elif agentPos == pos:
        B[pos] = 1
    #If we lost the signal we put it at the origin
    if B.totalCount() == 0:
      B[gameState.getInitialAgentPosition(self.index)] = 1
    B.normalize()
    self.B = B
    return self.B
    
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  believes = AgentBelieves()
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    
    self.believes.register(gameState, self.getOpponents(gameState))

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    Bs = self.believes.updateBelief(gameState.getAgentState(self.index).getPosition(), gameState)
#    self.displayDistributionsOverPositions(Bs)
    
    
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies_isPacman = [successor.getAgentState(i).isPacman for i in self.getOpponents(successor)]
    enemies_positions = [self.believes.getAgentPosition(i, successor) for i in self.getOpponents(successor)]
    invaders = [position for isPacman, position in zip(enemies_isPacman, enemies_positions) if isPacman]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, position) for position in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

