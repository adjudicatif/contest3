"""
Students' Names: Carpentier Jean
Contest Number: 3
Description of Bot: (a DETAILED description of your bot and it's strategy including any algorithms, heuristics, etc...)
"""
# myTeam.py
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
from game import Directions, Actions
import game
from util import nearestPoint
import capture
import numpy as np

from nn import *

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ReflexCaptureAgent', second = 'DefensiveReflexAgent'):
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
    
##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  believes = AgentBelieves()
  objectives = dict()
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    """MY CODE HERE"""
    self.believes.register(gameState, self.getOpponents(gameState))
    self.objectives[self.index] = None

    #define the border
    red = gameState.isOnRedTeam(self.index)
    x_border = max([x for x, _ in capture.halfGrid(gameState.getWalls(), red).asList()])
    self.border = [(x,y) for x, y in gameState.getWalls().asList(False) if x == x_border]

  def chooseAction(self, gameState):
    t = time.time()
    Bs = self.believes.updateBelief(gameState.getAgentState(self.index).getPosition(), gameState)
    self.displayDistributionsOverPositions(Bs)
    self.updateObjective(gameState)
    action = self.bestAction(gameState)
    print(time.time() - t)
    return action

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
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}




        