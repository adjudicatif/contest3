"""
Students' Names: Carpentier Jean
Contest Number: 3
Description of Bot:  thgough a shared markov chain between the two pacman
 - Bayes infer the positions
 - attempt to but a 200x200x6 neural network where the outputs are "aims" such as: eat closest food, flee from ghost 1, from ghost 2, hunt ghost1, hunt ghost2, bring back the food home
 - the features are the distances between the characters, the state variables of every player (numCarrying ,scaredTimer, isPacman) plus distances to food and to the border
 - the reward functions is: 1 for eating a pellet, 1 for bringing it back, 1 for eating an opponent with pellets, 1 for surviving with an enemy within 2 of distance and -0.1 for existing (to penalize useless actions)
 - training against baselineTeam
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
               first = 'SmartMultiTaskAgent', second = 'DefensiveReflexAgent'):
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
  def __init__(self, state_size, num_actions, learning_rate=0.1, gamma=0.97, batch_size=100, reward_penalty=0.01):
      self.num_actions = num_actions
      self.state_size = state_size
      self.learning_rate = learning_rate
      self.gamma = gamma
      self.batch_size = batch_size
      self.reward_penalty = reward_penalty
      self.records = []
      self.score = 0

      self.hs = [200, 200, self.num_actions]
      self.Ws = []
      self.bs = []
      i = self.state_size
      for h in self.hs:
        self.Ws.append(Variable(i, h))
        self.bs.append(Variable(h))
        i = h
       
      self.recover()
      self.save()
      
  def recover(self):
    try:
      with open("NN2", "rb") as f:
        d = pickle.Unpickler(f).load()
        if d["hs"] == self.hs:
          for W, W_matrix in zip(self.Ws, d["Ws"]):
            W.data = W_matrix
          for b, b_matrix in zip(self.bs, d["bs"]):
            b.data = b_matrix
          print("Recovered NN2")
        else:
          print("Erased NN2")
    except IOError:
      print("New NN2")
        
  def save(self):
    with open("NN2", "wb") as f:
      pickle.Pickler(f).dump({"Ws": [W.data for W in self.Ws], "bs": [b.data for b in self.bs], "hs": self.hs})
    
        
  def train(self, XS, YS):
    graph = self.run(XS, YS)
    graph.backprop()
    graph.step(self.learning_rate)
        
  def record(self, X, Y, A, R, score):
    assert not np.isnan(X).any()
    assert not np.isnan(Y).any()
    assert not np.isnan(A)
    assert not np.isnan(R)
    
    if len(self.records) < self.batch_size:
      self.records.append((X, Y, A, R))
    else: 
      self.recover()             
      X0 = np.array([X])
      Y0 = self.run(X0)
      Qtarget = 0#score - self.score
      self.score = score
      XS, YS = [], []
      Qtargets = []
      for X, Y, A, R in reversed(self.records):
        Qtarget = R + self.gamma * Qtarget
        Y[A] = Qtarget
        XS.append(X)
        YS.append(Y)
        Qtargets.append(Qtarget)
#      plt.plot(list(reversed(Qtargets)))
#      plt.show()
      XS, YS = np.array(XS), np.array(YS)
#      idx = np.random.choice(np.arange(self.batch_size), int(self.batch_size / 2), replace=False)
#      XS, YS = XS[idx], YS[idx]
      self.train(XS, YS)
      self.records = []
      self.save()
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

class SmartMultiTaskAgent(CaptureAgent):
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
    x_border = (not red)  + (gameState.getWalls().width / 2)#max([x for x, _ in capture.halfGrid(gameState.getWalls(), red).asList()])
    self.border = [(x,y) for x, y in gameState.getWalls().asList(False) if x == x_border]
    x_border_enemy = (red) + (gameState.getWalls().width / 2)
    self.border_enemy = [(x,y) for x, y in gameState.getWalls().asList(False) if x == x_border_enemy]
                   
    #define the neural net
    self.NN = DeepQModel(self.STATE_SIZE, len(self.AIMS), learning_rate=0.1)
    self.keep_on = 0
    self.max_depth = 5
  
  STATE_SIZE = 23
  def getState(self, gameState):
    team = [self.index, (self.index + 2) % 4]
    teamPos = [gameState.getAgentState(idx).getPosition() for idx in team]
    enemies = self.getOpponents(gameState)
    enemiesPos = [self.believes.getAgentPosition(idx, gameState) for idx in enemies]
    
    X = []
    
    #arePacman 4
    X += [gameState.getAgentState(idx).isPacman for idx in team + enemies]
    #scaredTimes 4
    X += [gameState.getAgentState(idx).scaredTimer / 100.0 for idx in team + enemies]
    #numCarrying 4
    X += [gameState.getAgentState(idx).numCarrying / 100.0 for idx in  team + enemies]
    #distToOther 1
    X += [self.getMazeDistance(teamPos[0], teamPos[1]) / 100.0]
    #distToEnemies 4
    X += [self.getMazeDistance(mate, enemy) / 100.0 for mate in teamPos for enemy in enemiesPos]
    #teamDistToBorder 2
    X += [min([self.getMazeDistance(mate, border) for border in self.border]) / 100.0 for mate in teamPos]
    #enemyDistToBorder 2
    X += [min([self.getMazeDistance(enemy, border) for border in self.border_enemy]) / 100.0 for enemy in enemiesPos]
    
    #distanceToFood 2
    foodList = self.getFood(gameState).asList()
    for mate in teamPos:
      if len(foodList) > 0: # This should always be True,  but better safe than sorry
        minDistance = min([self.getMazeDistance(mate, food) for food in foodList])  / 100.0 
      else:
        minDistance = 2
      X.append(minDistance)
    
    return np.array(X, dtype=float)
    
  AIMS = ["eat", "flee1", "flee2", "hunt1", "hunt2", "home"]
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
      
    def fleeFrom(myPos, enemyPos):
      dist = dict()
      possibleNextPos = []
      for action in gameState.getLegalActions(self.index):
        successor = self.getSuccessor(gameState, action)
        nextPos = successor.getAgentPosition(self.index)
        possibleNextPos.append(nextPos)
        if not self.isEaten(successor):
          dist[nextPos] = self.getMazeDistance(nextPos, enemyPos)
      if len(dist) > 0:
        nextPos = max(dist, key=dist.get) 
        assert nextPos != None
        return nextPos
      else:
        nextPos = random.choice(possibleNextPos)
        assert nextPos != None
        return nextPos    
        
    if aim == "flee1":
      myPos = gameState.getAgentState(self.index).getPosition()
      enemyPos = self.believes.getAgentPosition(self.getOpponents(gameState)[0], gameState)
      return fleeFrom(myPos, enemyPos)
          
    if aim == "flee2":
      myPos = gameState.getAgentState(self.index).getPosition()
      enemyPos = self.believes.getAgentPosition(self.getOpponents(gameState)[1], gameState)
      return fleeFrom(myPos, enemyPos)
      
    if aim == "hunt1":
      enemyPos = self.believes.getAgentPosition(self.getOpponents(gameState)[0], gameState)
      assert enemyPos != None
      return enemyPos
    
    if aim == "hunt2":
      enemyPos = self.believes.getAgentPosition(self.getOpponents(gameState)[1], gameState)
      assert enemyPos != None
      return enemyPos
      
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
    #self.displayDistributionsOverPositions(Bs)
    prevGameState = self.getPreviousObservation()
    team = [self.index, (self.index + 2) % 4]
    teamPos = [gameState.getAgentState(idx).getPosition() for idx in team]
    enemies = self.getOpponents(gameState)
    enemiesPos = [self.believes.getAgentPosition(idx, gameState) for idx in enemies]
    foodList = self.getFood(gameState).asList()
    if self.getPreviousObservation() != None:#after first step
      X = self.getState(self.getPreviousObservation())
      Y = self.NN.run(self.getState(self.getPreviousObservation()))
      A = self.aim_idx
      R = 0
      ### POS ###
      numEaten = gameState.getAgentState(self.index).numCarrying - prevGameState.getAgentState(self.index).numCarrying
      #for bringing back the food
      R += abs(numEaten) * 0.5 * (numEaten < 0 and teamPos[0] != gameState.getInitialAgentPosition(self.index))
      #for eating
      R += 0.5 * (numEaten > 0)
      #for hunting
      prevEnemiesPos = [self.believes.getAgentPosition(idx, prevGameState) for idx in enemies]
      prevDist = {enemyIdx: self.getMazeDistance(teamPos[0], enemy) for enemy, enemyIdx in zip(prevEnemiesPos, enemies)}
      enemyIdx = min(prevDist, key=prevDist.get)
      enemyPos = self.believes.getAgentPosition(enemyIdx, gameState)
      numEatenEnemy = gameState.getAgentState(enemyIdx).numCarrying - prevGameState.getAgentState(enemyIdx).numCarrying
      R += abs(numEatenEnemy) * 0.5 * (abs(numEatenEnemy) > 0  and prevDist[enemyIdx] <= 2 and enemyPos == gameState.getInitialAgentPosition(enemyIdx))
      #for surviving
      minDist = lambda gameState: min([self.getMazeDistance(teamPos[0], e) for e in self.believes.getbestAgentsPositions(gameState)])
      myIsPacman = gameState.getAgentState(self.index).isPacman
      R += self.NN.reward_penalty * (myIsPacman  and minDist(prevGameState) <= 2 and (minDist(gameState) -  minDist(prevGameState) >= 0))
      ### NEG ###
      #for being
      R -=  self.NN.reward_penalty
      #for being eaten: lose of the food
      R -= 0.5 * (1 + abs(numEaten > 0)) * (teamPos[0] == gameState.getInitialAgentPosition(self.index))
      self.NN.record(X, Y, A, R, self.getScore(gameState))
      
    #Split the actors
    if teamPos[0] == gameState.getInitialAgentPosition(self.index):
        print("restart")
        self.keep_on = 0
        self.aim_idx = self.NN.get_action(self.getState(gameState), eps=0)
    #Keep on
    elif self.keep_on > 0:
      self.keep_on -= 1
      print("keep_on", self.keep_on)
        
    #TRAIN
    if True:
      #NEXT BEHAVIOR
      if self.keep_on > 0: pass
      #Free will
      elif util.flipCoin(0.95):
        print("choice")
        self.aim_idx = self.NN.get_action(self.getState(gameState), eps=0.2)
      #Behavior to force
      elif util.flipCoin(0.16) or (util.flipCoin(0.5) and gameState.getAgentState(self.index).numCarrying > 0):
        self.aim_idx = 5
        self.keep_on = min([self.getMazeDistance(teamPos[0], home) for home in self.border])
        print("force home", self.keep_on)
      elif util.flipCoin(0.16) or (util.flipCoin(0.5) and 5 > self.getMazeDistance(teamPos[0], enemiesPos[0])):
        self.aim_idx = 1
        self.keep_on = 5
        print("force flee1", self.keep_on)
      elif util.flipCoin(0.16) or (util.flipCoin(0.5) and 5 > self.getMazeDistance(teamPos[0], enemiesPos[1])):
        self.aim_idx = 2
        self.keep_on = 5
        print("force flee2", self.keep_on)      
      elif util.flipCoin(0.16) or (util.flipCoin(0.5) and 0 < gameState.getAgentState(enemies[0]).numCarrying):
        self.aim_idx = 3
        self.keep_on = self.getMazeDistance(teamPos[0], enemiesPos[0])
        print("force hunt1", self.keep_on)
      elif util.flipCoin(0.16) or (util.flipCoin(0.5) and 0 < gameState.getAgentState(enemies[1]).numCarrying):
        self.aim_idx = 4
        self.keep_on = self.getMazeDistance(teamPos[0], enemiesPos[1])
        print("force hunt2", self.keep_on)
      else:
        self.aim_idx = 0
        if len(foodList) > 2:
          self.keep_on = min([self.getMazeDistance(teamPos[0], food) for food in foodList])
        print("force food", self.keep_on)
    #USE
    else:
      self.aim_idx = self.NN.get_action(self.getState(gameState), eps=0)
    
    aim = self.AIMS[self.aim_idx]
    action = self.getActionToObjective(self.getObjective(aim, gameState), gameState)
    print(R)
    print(self.getState(gameState).round(2))
    print(np.round(self.NN.run(self.getState(gameState)), 3), self.AIMS[self.aim_idx], self.isEaten(gameState), action)#, self.getObjective(aim, gameState), action, )
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


class DefensiveReflexAgent(SmartMultiTaskAgent):
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


        