"""
Students' Names: Jean Carpentier
Contest Number: 3
Description of Bot:
  - checks if the pacman enters a trap ie if it is eatable. A trap would be a corridor without any exit so that a hunting oppponent would eat it for sure.
  - 1 defenser: determines the best place to go to defend on the border between the two worlds
  - 1 attacker: tries to reach the first pellet on a "safe" way without other's
  - Bayes inference for the position of the opponents
"""

from captureAgents import CaptureAgent
import baselineTeam
import random, time, util
from game import Directions, Actions
import game
import distanceCalculator
import capture

#class Tree:
#  def __init__(self):

################
# Shared data #
#################
class SharedData:
  #Parmaeters
  halfhalf = False
  
  def __init__(self, isRed):
      self.agents = []
      self.isRed = isRed
      self.color = None
      self.searched = False
      self.registered = False
      self.distancer = None
      self.mazeDistance = None
      self.objectives = dict()
      
      
  def registerInitialState(self, gameState):
      self.distancer = distanceCalculator.Distancer(gameState.data.layout)
      self.distancer.getMazeDistances() #searches all paths, done once AT ALL
      self.mazeDistance = self.distancer.getDistance #function
      self.splitTheFood(self.getFood(gameState), halfhalf=self.halfhalf)
      
      self.home = set(capture.halfList(gameState.getWalls().asList(False), gameState.getWalls(), self.isRed))
      x_border = (not self.isRed)  + gameState.getWalls().width / 2
      self.border = [(x,y) for x, y in gameState.getWalls().asList(False) if x == x_border]
      self.border_enemy = {(x,y) for x, y in gameState.getWalls().asList(False) if x == self.isRed  + gameState.getWalls().width / 2}
      print(self.border, self.home)
      self.registered = True
      print "registered", self.isRed
          
  #### FOOD TOPICS
  def splitTheFood(self, foodList, halfhalf=True):
      "Random split"
      foodSet = [set(), set()]
      for food in foodList:
        if halfhalf: foodSet[int(util.flipCoin(0.5))].add(food)
        else: 
          foodSet[0].add(food)
          foodSet[1].add(food)
      self.foodSet = {self.agents[i]: foodSet[i] for i in range(2)}
      print "food splitted", self.foodSet
  def getFood(self, gameState):
      "return as List // need to aim the OTHER's food"
      if self.isRed: return gameState.getBlueFood().asList()
      else: return gameState.getRedFood().asList()
  def getMyObjective(self, agent, gameState):
    "Method = the closest"
#        print "METH: getMyObjective", agent
    pos = gameState.getAgentPosition(agent)
    foodAvailable = set(self.getFood(gameState))
    myFood = foodAvailable.intersection(self.foodSet[agent])
    if len(myFood) > 0: 
      return self.closest(pos, myFood)
    else: return self.closest(pos, self.home)#Go back home
        
  def closest(self, ori, dests):
    aMIN, MIN = None, 1e10
    for dest in dests:
      d = self.mazeDistance(ori, dest) 
      if d < MIN: 
          aMIN, MIN = dest, d
    return aMIN
            
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DefensiveReflexAgent', second = 'EatAgent'):
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
    #The following line is an example only; feel free to change it.
    shared = SharedData(isRed)
    firstAgent = EatAgent(firstIndex)
    firstAgent.shared = shared
    secondAgent = EatAgent(secondIndex)
    secondAgent.shared = shared
    #Now we update shared data
    shared.agents = [firstIndex, secondIndex]
    shared.roles = {firstIndex: 'eat', secondIndex:'defend'}
    #We need to initalize the distances
    return [firstAgent, secondAgent]
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
    
#########
# TOOLS #
#########
class MyBasicAgent(baselineTeam.ReflexCaptureAgent):
  """
  Used for adding extra variables
  """
  believes = AgentBelieves()
  def registerInitialState(self, gameState):
    """
    To let this like that
    """
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()
    '''
    Your initialization code goes here, if you need any.
    '''
    self.shared.registerInitialState(gameState)
    self.believes.register(gameState, self.getOpponents(gameState))
    print("Register C2", self.index)
    
  def closest(self, ori, dests):
    aMIN, MIN = None, 1e10
    for dest in dests:
      d = self.getMazeDistance(ori, dest) 
      if d < MIN: 
          aMIN, MIN = dest, d
    return aMIN
    
  def simulateFlee(self, agentPos, enemyPos, walls, depth=5, visited={}):
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
    

#  def simulateFlee(self, agentPos, enemyPos, walls, depth=5):
#    "The enemy plays first"
##    print agentPos, enemyPos
#    positions = {pos: self.getMazeDistance(agentPos, pos) for pos in game.Actions.getLegalNeighbors(enemyPos, walls)}
#    nextEnemyPos = min(positions, key=positions.get)
#    if depth == 0 or agentPos == nextEnemyPos:#so that detects if eaten
#      return agentPos, nextEnemyPos
#    distances = {}
#    alpha = 0 #self.getMazeDistance(agentPos, enemyPos)
#    for nextAgentPos in game.Actions.getLegalNeighbors(agentPos, walls):
#      d = self.getMazeDistance(nextAgentPos, nextEnemyPos)
##      if d < alpha: continue
#      alpha = max(d, alpha)
#      lastAgentPos, lastEnemyPos = self.simulateFlee(nextAgentPos, nextEnemyPos, walls, depth - 1)
#      distances[(lastAgentPos, lastEnemyPos)] = self.getMazeDistance(lastAgentPos, lastEnemyPos)
#    return max(distances, key=distances.get)
    
  def isEatable(self, gameState, eaters, max_depth=5):
    myPos = gameState.getAgentPosition(self.index) 
    for enemyPos in [self.believes.getAgentPosition(enemy, gameState) for enemy in eaters]:
      finalAgent, finalEnemy = self.simulateFlee(myPos, enemyPos, gameState.getWalls(), max_depth)
      if finalAgent == finalEnemy: return 1 #one suficces
    return 0
    
  def getFoodObjective(self, gameState):
    def closestSafe(agent, objsList):
      myPos = gameState.getAgentPosition(self.index)
      foods = {food: self.getMazeDistance(myPos, food) for food in objsList}
      foods = sorted(foods, key=foods.get)
      for food in foods[:-2]:
        if self.canReachSafely(gameState, food, 5):
          return food
      return None
          
    foods = self.getFood(gameState).asList()
    closest_food_safe = closestSafe(self.index, foods)
    if closest_food_safe != None: return closest_food_safe
    if len(foods) > 2: return foods[0] #still some work yet, take one
    closest_border_safe = closestSafe(self.index, self.shared.border)
    if closest_border_safe != None: return closest_border_safe
    return self.shared.border[0]
    
  def getObjective(self, gameState):
    if self.shared.roles[self.index] == "eat":
      assert self.getFoodObjective(gameState) != None
      return self.getFoodObjective(gameState)
      
    elif self.shared.roles[self.index] == "defend":
      getPosition = lambda idx: self.believes.getAgentPosition(idx, gameState)
      myPos = gameState.getAgentPosition(self.index)
      states = {idx: gameState.getAgentState(idx) for idx in self.getOpponents(gameState)}
      carrier = {idx: s.numCarrying for idx, s in states.items()}
      homes = {idx: self.closest(getPosition(idx), self.shared.border) for idx in states}

      #If we are the enemy is close from the outside: rush on HIM
      enemy_to_home = {idx: self.getMazeDistance(getPosition(idx), homes[idx]) for idx in states}
      for idx in sorted(enemy_to_home, key=enemy_to_home.get):
        if enemy_to_home[idx] < 3 and carrier[idx] > 0:
          assert getPosition(idx) != None
          return getPosition(idx)
      
      #If the enemy is still far, just go there and wait
      me_to_home = {idx: self.getMazeDistance(myPos, homes[idx]) for idx in states}
      reachable = {idx: carrier[idx] for idx in states if me_to_home[idx] < enemy_to_home[idx]} 
      for idx in sorted(reachable, key=reachable.get):
        if carrier[idx] > 0:
          assert homes[idx] != None
          return homes[idx]
       
      #if nobody carries anything    
      assert getPosition(min(me_to_home, key=me_to_home.get)) != None
      return getPosition(min(me_to_home, key=me_to_home.get))

  def canReachSafely(self, gameState, food, max_depth):
    def goTo(ori, dest, dt):
      if dt == 0: return ori
      d = dict()
      for nextPos in game.Actions.getLegalNeighbors(ori, gameState.getWalls()):
        d[nextPos] = self.getMazeDistance(nextPos, dest)
      return sorted(d, key=d.get)[0]

    myPos = gameState.getAgentPosition(self.index)
    
    isPacman = {idx: gameState.getAgentState(idx).isPacman for idx in self.getOpponents(gameState)}
    enemiesPos = [self.believes.getAgentPosition(idx, gameState) for idx  in self.getOpponents(gameState) if not isPacman[idx]]#filter if attacks
    
    dt = min(max_depth, self.getMazeDistance(myPos, food))
    nextMyPos = goTo(myPos, food, dt)
    
    for enemyPos in enemiesPos:
      nextEnemyPos = goTo(enemyPos, food, dt)
      #if we got closer
      if self.getMazeDistance(nextMyPos, nextEnemyPos) < self.getMazeDistance(myPos, enemyPos):
        #if he is closer than me
        if self.getMazeDistance(nextMyPos, food) > self.getMazeDistance(nextEnemyPos, food):
          return False
    return True

  def chooseAction(self, gameState, kind="myopic", depth=3):
    """
    CALLED BY THE GAME
    """
#    print ""
    Bs = self.believes.updateBelief(gameState.getAgentState(self.index).getPosition(), gameState)
    self.displayDistributionsOverPositions(Bs)
    
    self.shared.objectives[self.index] = self.getObjective(gameState)
    score, action = self.chooseActionMyopic(gameState)
#    print "CHOICE myopic:", self.index, action, score
    return action

  def chooseActionMyopic(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    action = random.choice(bestActions)
    return maxValue, action
    
    
##########
# Agents #
##########

class EatAgent(MyBasicAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = successor.getAgentPosition(self.index)
    prevPos = gameState.getAgentPosition(self.index)
    
    #SCORE = NB pellets
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = - len(foodList)
    
    # GO TO OBJ
    objPos = self.getObjective(successor)
    if self.shared.objectives[self.index] != objPos:
      features["changeObjective"] = 1
    objDist = self.getMazeDistance(myPos, objPos)
    features['distanceToObjective'] = objDist
    
    
    def is_dangerous(idx):
      state = successor.getAgentState(idx)
      distance = self.getMazeDistance(myPos, self.believes.getAgentPosition(idx, successor)) #not distance in 
      #ME Vs OTHER
      ghostScaredVsPacman = (not myState.isPacman) and myState.scaredTimer > distance and state.isPacman 
      pacmanVsActiveGhost = myState.isPacman and state.scaredTimer < distance and (not state.isPacman)
      return ghostScaredVsPacman or pacmanVsActiveGhost
      
    #STATE
    ###########################################################################
    if self.shared.roles[self.index] == "defend":
      features['onDefense'] = 1
      if myState.isPacman: features['onDefense'] = 0

      #Computes distance to invaders we can see
      enemies = [self.believes.getAgentPosition(idx, successor) for idx in self.getOpponents(successor)]
      arePacman = [successor.getAgentState(idx).isPacman for idx in self.getOpponents(successor)]
      invaders = [1 for isPacman, position  in zip(enemies, arePacman) if isPacman and position != None]
      features['numInvaders'] = len(invaders)
      
      if action == Directions.STOP: features['stop'] = 1
      rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      if action == rev: features['reverse'] = 1
  
      #EATABLE
      max_depth = 5
      dangerous_enemies = [idx for idx in self.getOpponents(successor) if is_dangerous(idx)]
      features["eatable"] = self.isEatable(successor, dangerous_enemies, max_depth)
      
      
    ############################################################################
    else:
      # NUM CARRYING
      features['numCarrying'] = successor.getAgentState(self.index).numCarrying
  
      distances = {dest: self.getMazeDistance(myPos, dest) for dest in self.shared.home}
      homePos = sorted(distances, key=distances.get)[0]
      homeDist = self.getMazeDistance(myPos, homePos)
      homeObj = self.getMazeDistance(homePos, objPos)
      
      #implicitly make it tend to home to get back the pellet
      risk_aversion = 2 / 5 * features['numCarrying']
      if features['numCarrying'] > 0 and homeDist + homeObj < risk_aversion * objDist: 
        features['distanceToObjective'] = homeDist
        
      
      #AVOID TO DIE: if the enemy is far: don't care
      max_depth = 5
      dangerous_enemies = [idx for idx in self.getOpponents(successor) if is_dangerous(idx)]
      if len(dangerous_enemies) > 0:
        eatable = 0
        eatable_distances = {}
        for enemyPos in [self.believes.getAgentPosition(enemy, successor) for enemy in dangerous_enemies]:
          finalAgent, finalEnemy = self.simulateFlee(myPos, enemyPos, successor.getWalls(), max_depth)
          eatable_distances[finalEnemy] = self.getMazeDistance(myPos, finalEnemy)
          if finalAgent == finalEnemy and finalAgent not in self.shared.home: 
            eatable = 1 #one suficces
        features["eatable"] = eatable
        features["eatableDistance"] = eatable * min(eatable_distances.values())
#    print action, objPos
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'numCarrying':-50, 'changeObjective':-2, 
            'numInvaders': -1000, 'onDefense': 100, 'stop': -100, 'reverse': -2,
            'eatable':-1000, 'eatableDistance':10 ,'distanceToObjective': -1}
            
    