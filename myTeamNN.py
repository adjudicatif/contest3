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

import pickle
import matplotlib.pyplot as plt

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
    
class DeepQModel:
  """
  Bla
  """
  def __init__(self, state_size, num_actions, learning_rate=0.05, gamma=0.9, batch_size=50, reward_penalty=0.05):
      self.num_actions = num_actions
      self.state_size = state_size
      self.learning_rate = learning_rate
      self.gamma = gamma
      self.batch_size = batch_size
      self.reward_penalty = reward_penalty
      self.records = []
      self.score = 0

      self.hs = [100, 100, self.num_actions]
      self.Ws = []
      self.bs = []
      i = self.state_size
      for h in self.hs:
        self.Ws.append(Variable(i, h))
        self.bs.append(Variable(h))
        i = h
       
      try:
        with open("NN", "rb") as f:
          d = pickle.Unpickler(f).load()
          if d["hs"] == self.hs:
            for W, W_matrix in zip(self.Ws, d["Ws"]):
              W.data = W_matrix
            for b, b_matrix in zip(self.bs, d["bs"]):
              b.data = b_matrix
            print("Recovered NN")
          else:
            print("Erased NN")
      except IOError:
        print("New NN")
        
  def train(self, XS, YS):
    graph = self.run(XS, YS)
    graph.backprop()
    graph.step(self.learning_rate)
        
  def record(self, X, Y, A, R, score):
      if len(self.records) < self.batch_size:
        self.records.append((X, Y, A, R))
      else:
        X0 = np.array([X])
        Y0 = self.run(X0)
        Qmax = score - self.score
        self.score = score
        XS, YS = [], []
        for X, Y, A, R in reversed(self.records):
          Qmax = R + self.gamma * Qmax
          Y[A] = Qmax
          XS.append(X)
          YS.append(Y)
        XS, YS = np.array(XS), np.array(YS)
        self.train(XS, YS)
        self.records = []
#        plt.plot(np.flipud(YS))
#        plt.show()
        with open("NN", "wb") as f:
          pickle.Pickler(f).dump({"Ws": [W.data for W in self.Ws], "bs": [b.data for b in self.bs], "hs": self.hs})
        print("update", np.round(Y0, 3), np.round(self.run(X0), 3))
          
  def run(self, states, Q_target=None):
      graph = Graph(self.Ws + self.bs)
      X = Input(graph, states)
      for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
        XW = MatrixMultiply(graph, X, W)
        XW_plus_b = MatrixVectorAdd(graph, XW, b)
        if i < len(self.Ws) - 1:
          X = ReLU(graph, XW_plus_b)
        else:
          X = XW_plus_b
      if Q_target is not None:
          Y = Input(graph, Q_target)
          loss = SquareLoss(graph, X, Y)
          return graph
      else:
          return graph.get_output(X)

  def get_action(self, state, eps):
      """
      Select an action for a single state using epsilon-greedy.

      Inputs:
          state: a (1 x 4) numpy array
          eps: a float, epsilon to use in epsilon greedy
      Output:
          the index of the action to take (either 0 or 1, for 2 actions)
      """
      if np.random.rand() < eps:
          return np.random.choice(self.num_actions)
      else:
          scores = self.run(state)
          return int(np.argmax(scores))
          
  
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
    print("NN is registered", self.index)
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    """MY CODE HERE"""
    #
    self.believes.register(gameState, self.getOpponents(gameState))
    self.objectives[self.index] = None

    #define the border
    red = gameState.isOnRedTeam(self.index)
    x_border = (not red)  + gameState.getWalls().width / 2#max([x for x, _ in capture.halfGrid(gameState.getWalls(), red).asList()])
    self.border = [(x,y) for x, y in gameState.getWalls().asList(False) if x == x_border]
                   
    #define the neural net
    self.NN = DeepQModel(self.STATE_SIZE, len(self.AIMS), learning_rate=0.1)
    self.keep_on = 0
    self.max_depth = 5
  
  STATE_SIZE = 11
  def getState(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition() 
    dist = {idx: self.getMazeDistance(self.believes.getAgentPosition(idx, gameState), myPos) for idx in self.getOpponents(gameState)}
    sorted_enemies = sorted(dist, key=dist.get)
    X = []
    
    #numCarrying
    X.append(gameState.getAgentState(self.index).numCarrying  / 100.0)
    #isPacman
    X.append(gameState.getAgentState(self.index).isPacman)
    #scaredTime
    X.append(gameState.getAgentState(self.index).scaredTimer  / 100.0)
    #distanceToFood
    foodList = self.getFood(gameState).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) / 100.0 for food in foodList])
    else:
      minDistance = 1e2
    X.append(minDistance)
    #distanceToHome
    X.append(min([self.getMazeDistance(pos, myPos) / 100.0 for pos in self.border]))
    #distanceToEnemies
    X += [self.getMazeDistance(self.believes.getAgentPosition(idx, gameState), myPos) / 100.0 for idx in sorted_enemies]
    #arePacman
    X += [gameState.getAgentState(idx).isPacman for idx in sorted_enemies]
    #scaredTimes
    X += [gameState.getAgentState(idx).scaredTimer / 100.0 for idx in sorted_enemies]
          
    return np.array(X, dtype=float)
    
  AIMS = ["eat", "flee", "home"]
  def getObjective(self, aim, gameState):
    if aim not in self.AIMS: aim == "home"
    if len(self.getFood(gameState).asList()) <= 2: aim == "home"
      
    if aim == "eat":
      myPos = gameState.getAgentState(self.index).getPosition()
      foodList = self.getFood(gameState).asList()
      if len(foodList) > 2: # This should always be True,  but better safe than sorry
        dist = {food: self.getMazeDistance(myPos, food) for food in foodList}
        assert min(dist, key=dist.get) != None
        return min(dist, key=dist.get)
      else:
        print("less than 2 food")
        aim = "home"
      
    if aim == "flee":
      myPos = gameState.getAgentState(self.index).getPosition()
      enemies = [pos for pos in self.believes.getbestAgentsPositions(gameState)]
      dist = dict()
      for action in gameState.getLegalActions(self.index):
        successor = self.getSuccessor(gameState, action)
        nextPos = successor.getAgentPosition(self.index)
        if not self.isEaten(successor):
          assert nextPos != None
          return nextPos
        dist[nextPos] = min([self.getMazeDistance(nextPos, enemy) for enemy in enemies])
      print("is eaten, flees", dist)
      nextPos = max(dist, key=dist.get) 
      assert nextPos != None
      return nextPos
      
      
    if aim == "home":
      myPos = gameState.getAgentState(self.index).getPosition()
      dist = {home: self.getMazeDistance(myPos, home) for home in self.border}
      assert min(dist, key=dist.get) != None
      return min(dist, key=dist.get)
      
    raise NotImplementedError
    
  def getActionToObjective(self, objective, gameState):
    assert objective != None
    dist = {}
    for action in gameState.getLegalActions(self.index):
      successor = self.getSuccessor(gameState, action)
      nextPos = successor.getAgentPosition(self.index)
      dist[action] = self.getMazeDistance(nextPos, objective)
    return min(dist, key=dist.get)   

  def chooseAction(self, gameState):
    R = 0
    self.believes.updateBelief(gameState.getAgentState(self.index).getPosition(), gameState)
#    self.displayDistributionsOverPositions(Bs)

    #TRAIN
    if False:
      #LEARNING
      prevGameState = self.getPreviousObservation()
      if self.getPreviousObservation() != None:#after first step
        X = self.getState(self.getPreviousObservation())
        Y = self.NN.run(self.getState(self.getPreviousObservation()))
        A = self.aim_idx
        R = 0
        myPos = gameState.getAgentState(self.index).getPosition()
        numEaten = gameState.getAgentState(self.index).numCarrying - prevGameState.getAgentState(self.index).numCarrying
        #for bringing back the food
        R += (1.0 + abs(numEaten) / 10.0) * (numEaten < 0 and myPos != gameState.getInitialAgentPosition(self.index))
        #for eating
        R += 1.0 * (numEaten > 0)
        #for surviving
        minDist = lambda gameState: min([self.getMazeDistance(myPos, e) for e in self.believes.getbestAgentsPositions(gameState)])
        R += 1.0 * (minDist(prevGameState) <= 2 and minDist(gameState) >= 2)
#        R += -(1.0 + abs(numEaten) / 10.0) * (myPos == gameState.getInitialAgentPosition(self.index))
        #for being
        R -=  self.NN.reward_penalty
        self.NN.record(X, Y, A, R, self.getScore(gameState))
        
      #NEXT BEHAVIOR
      if gameState.getAgentState(self.index).getPosition() == gameState.getInitialAgentPosition(self.index):
        print("restart")
        self.keep_on = 0
        self.aim_idx = self.NN.get_action(self.getState(gameState), eps=0)
      elif self.keep_on > 0:
        self.keep_on -= 1
      #Behavior to force
      elif util.flipCoin(0.1) and gameState.getAgentState(self.index).numCarrying > 0:
        self.aim_idx = 2
        myPos = gameState.getAgentState(self.index).getPosition()
        self.keep_on = min([self.getMazeDistance(myPos, home) for home in self.border])
        print("force home", self.keep_on)
      elif util.flipCoin(0.5) and 5 > min([self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), 
                                       enemy) for enemy in self.believes.getbestAgentsPositions(gameState)]):
        self.aim_idx = 1
        self.keep_on = 5
        print("force flee", self.keep_on)
      #Random behavior
      elif util.flipCoin(0.05):
        if util.flipCoin(0.6):
          self.aim_idx = 0
          myPos = gameState.getAgentState(self.index).getPosition()
          foodList = self.getFood(gameState).asList()
          if len(foodList) > 2:
            self.keep_on = min([self.getMazeDistance(myPos, food) for food in foodList])
          print("flip food", self.keep_on)
        elif util.flipCoin(0.5):
          self.aim_idx = 2
          myPos = gameState.getAgentState(self.index).getPosition()
          self.keep_on = min([self.getMazeDistance(myPos, home) for home in self.border])
          print("flip home", self.keep_on)
        else:
          self.aim_idx = 1
          self.keep_on = 5
          print("flip flee", self.keep_on)
      else:
        print("choice")
        self.aim_idx = self.NN.get_action(self.getState(gameState), eps=0)
    #USE
    else:
      self.aim_idx = self.NN.get_action(self.getState(gameState), eps=0)
    
    aim = self.AIMS[self.aim_idx]
    action = self.getActionToObjective(self.getObjective(aim, gameState), gameState)
    print(R, np.round(self.getState(gameState), 3), np.round(self.NN.run(self.getState(gameState)), 3), self.AIMS[self.aim_idx], self.isEaten(gameState))#, self.getObjective(aim, gameState), action, )
#    print(time.time() - t)
    return action
    

  def simulateFlee(self, agentPos, enemyPos, walls, depth, visited={}):
    """
    Returns two identical position if flee not possible (random positions ! because of visited)
    Returns the solution at max depth after corridor
    """
    positions = {pos: self.getMazeDistance(agentPos, pos) for pos in game.Actions.getLegalNeighbors(enemyPos, walls)}
    nextEnemyPos = min(positions, key=positions.get)
    if depth == 0 or agentPos == nextEnemyPos:#so that detects if eaten
      return agentPos, nextEnemyPos
      
    distances = {}
    nextPositions = [p for p in game.Actions.getLegalNeighbors(agentPos, walls) if p not in visited]
    nextDepth = depth - int(len(nextPositions) > 1)
    for nextAgentPos in nextPositions:
      nextVisited = set(visited)
      nextVisited.add(nextAgentPos)
      lastAgentPos, lastEnemyPos = self.simulateFlee(nextAgentPos, nextEnemyPos, 
                                                     walls, nextDepth, nextVisited)
      distances[(lastAgentPos, lastEnemyPos)] = self.getMazeDistance(lastAgentPos, lastEnemyPos)
    if len(distances) == 0:
      return enemyPos, enemyPos
    return max(distances, key=distances.get)
    
  def isEaten(self, gameState):
    max_depth = self.max_depth
    
    eaters = [enemy for enemy in self.getOpponents(gameState) if not gameState.getAgentState(enemy).isPacman]
    myPos = gameState.getAgentPosition(self.index) 
    for enemyPos in [self.believes.getAgentPosition(enemy, gameState) for enemy in eaters]:
      if myPos == gameState.getInitialAgentPosition(self.index):
        return True
      finalAgent, finalEnemy = self.simulateFlee(myPos, enemyPos, gameState.getWalls(), max_depth)
      if finalAgent == finalEnemy: 
        return True #one suficces
    return False
    
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

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
    
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


        