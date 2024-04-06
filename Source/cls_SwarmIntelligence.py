from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

import cls_ObjectiveFunctions
from cls_GeneralFunctions import GeneralFunctions

# Visualization (Plotly)
import plotly.graph_objs as go
from plotly.subplots import make_subplots


#region ------------------- Particle Swarm Optimizer (PSO) -------------------

#region PSO Results
@dataclass
class PSOresult:

    """Class for returning result of GWO algorithm"""
    _NumberOfParticles: int
    _NumberOfDimensions: int
    _NumberOfIterations: int

    _ObjectiveFunctionNumberOfCalls: int
    _ObjectiveFunction: cls_ObjectiveFunctions.ObjectiveFunction

    _GlobalBestObjectiveAllIterations: np.array
    _GlobalBestPositionAllIterations: np.array
    _PersonalBestObjectiveAllIterations: np.array
    _PersonalBestPositionAllIterations: np.array

    _PositionsAllIterations: np.array
    _WeightAll: np.array

    # region [FUNCTIONS] INIT()
    def __init__(self, NumberOfParticles: int, NumberOfDimensions: int, NumberOfIterations: int,
                 ObjectiveFunctionNumberOfCalls: int, ObjectiveFunction: cls_ObjectiveFunctions.ObjectiveFunction,
                 GlobalBestObjectiveAllIterations: np.array, GlobalBestPositionAllIterations: np.array,
                 PersonalBestObjectiveAllIterations: np.array, PersonalBestPositionAllIterations: np.array,
                 PositionAllIterations: np.array, WeightHistory: np.array):

        # if not isinstance(ObjectiveFunction, cls_ObjectiveFunctions.ObjectiveFunction):
        #     raise Exception
        # if not isinstance(aFunction, GWOaFunctionsList):
        #     raise Exception
        self._ObjectiveFunction = ObjectiveFunction
        self._NumberOfParticles = NumberOfParticles
        self._NumberOfDimensions = NumberOfDimensions
        self._NumberOfIterations = NumberOfIterations
        self._ObjectiveFunctionNumberOfCalls = ObjectiveFunctionNumberOfCalls
        self._GlobalBestObjectiveAllIterations = GlobalBestObjectiveAllIterations
        self._GlobalBestPositionAllIterations = GlobalBestPositionAllIterations
        self._PersonalBestObjectiveAllIterations = PersonalBestObjectiveAllIterations
        self._PersonalBestPositionAllIterations = PersonalBestPositionAllIterations
        self._PositionsAllIterations = PositionAllIterations
        self._WeightAll = WeightHistory

    # endregion

    # region [FUNCTIONS]
    def printData(self):
        str = f"Number of Particles: {self._NumberOfParticles}\nNumber of Dimensions: {self._NumberOfDimensions} \nNumber of Iterations: {self._NumberOfIterations}\n" \
              f"Objective Function: {self._ObjectiveFunction}\nNumber of Objective Function Calls: {self._ObjectiveFunctionNumberOfCalls}\n" \
              f"Global best objective: {self._GlobalBestObjectiveAllIterations[-1]}\n" \
              f"Global best position: {self._GlobalBestPositionAllIterations[-1]}"
        return str


    def GetPositionsHistoryReducedDimension(self, particleNum, toDimension):
        return GeneralFunctions.getReducedDimensionByPCA(self._PositionsAllIterations[:,particleNum], toDimension)

    def GetBestPositionReducedDimension(self, toDimension):
        return GeneralFunctions.getReducedDimensionByPCA(self._GlobalBestPositionAllIterations, toDimension)

    # endregion

    #region ATTRIBUTES [SET & GET]

    def _get_NumberOfParticles(self):
        return self._NumberOfParticles
    def _set_NumberOfParticles(self, value):
        self._NumberOfParticles = value
    numberOfParticles = property(fget=_get_NumberOfParticles, fset=_set_NumberOfParticles, doc="The Number of Particles")

    def _get_NumberOfDimensions(self):
        return self._NumberOfDimensions
    def _set_NumberOfDimensions(self, value):
        self._NumberOfDimensions = value
    numberOfDimensions = property(fget=_get_NumberOfDimensions, fset=_set_NumberOfDimensions, doc="The Number of Dimensions")

    def _get_NumberOfIterations(self):
        return self._NumberOfIterations
    def _set_NumberOfIterations(self, value):
        self._NumberOfIterations = value
    numberOfIterations = property(fget=_get_NumberOfIterations, fset=_set_NumberOfIterations, doc="The Number of Iterations")

    def _get_ObjectiveFunction(self):
        return self._ObjectiveFunction
    def _set_ObjectiveFunction(self, value):
        self._ObjectiveFunction = value
    objectiveFunction = property(fget=_get_ObjectiveFunction, fset=_set_ObjectiveFunction, doc="The Objective Function")

    def _get_ObjectiveFunctionNumberOfCalls(self):
        return self._ObjectiveFunctionNumberOfCalls
    objectiveFunctionNumberOfCalls = property(fget=_get_ObjectiveFunctionNumberOfCalls, doc="The number of calls for Objective Function")

    def _get_PositionsAllIteration(self):
        return self._PositionsAllIterations
    positionsAllIteration = property(fget=_get_PositionsAllIteration, doc="Positions array of all particles in all iterations")

    def _get_GlobalBestObjectiveAllIterations(self):
        return self._GlobalBestObjectiveAllIterations
    globalBestObjectiveAllIteration = property(fget=_get_GlobalBestObjectiveAllIterations, doc="Global best fitness array in all iterations")

    def _get_GlobalBestPositionAllIterations(self):
        return self._GlobalBestPositionAllIterations
    globalBestPositionAllIteration = property(fget=_get_GlobalBestPositionAllIterations, doc="Global best position array in all iterations")

    def _get_PersonalBestObjectiveAllIterations(self):
        return self._PersonalBestObjectiveAllIterations
    personalBestObjectiveAllIteration = property(fget=_get_PersonalBestObjectiveAllIterations, doc="Personal best fitness array in all iterations")

    def _get_PersonalBestPositionAllIterations(self):
        return self._PersonalBestPositionAllIterations
    personalBestPositionAllIteration = property(fget=_get_PersonalBestPositionAllIterations, doc="Personal best position array in all iterations")

    def _get_WeightsAll(self):
        return self._WeightAll
    weightsAll = property(fget=_get_WeightsAll, doc="Weights array in all iterations")

    #endregion

#endregion

#region PSO and its Variants

class PSO():

    def PSO(self, iterationCount, particlesNo, nVar, selectedObjectiveFunction, weightMax, weightMin,
            velocityCtrlCoeff, toMinimize, cParam1 = 2, cParam2 = 2, trainX=None, trainY=None, testX=None,
            testY=None):

        #region 1. Reading inputs and set the attributes

        self._particlesNo = particlesNo  # The number of search agents
        self._nVar = nVar  # The number of parameters or dimensions
        self._iterationCount = iterationCount
        self._weightMax = weightMax
        self._weightMin = weightMin
        self._velocityCtrlCoeff = velocityCtrlCoeff
        self._toMinimize = toMinimize
        self._selectedObjectiveFunctionClass = selectedObjectiveFunction
        self._selectedObjectiveFunction = selectedObjectiveFunction()

        # -------------- Reading required parameters from the selected function --------------
        bounds = self._selectedObjectiveFunction.functionBoundary
        lb = bounds[0]
        ub = bounds[1]
        lowerBound = lb * np.ones(self._nVar)
        upperBound = ub * np.ones(self._nVar)
        boundaryNo = upperBound.shape[0]  # OR lowerBound (No difference)

        c1 = cParam1 # 0.5
        c2 = cParam2  # 0.3
        velocityMax = (ub - lb) * self._velocityCtrlCoeff
        velocityMin = -velocityMax

        #endregion

        # region 2. [PSO-related Parameters]

        if self._toMinimize:
            val = np.inf
        else:
            val = -np.inf

        _particlesPosition = np.random.random((self._particlesNo, self._nVar)) * (upperBound - lowerBound) + lowerBound
        _particlesVelocity = np.zeros((self._particlesNo, self._nVar))
        _particleObjective = np.ones(self._particlesNo) * val
        _particlePersonalBestPosition = np.zeros((self._particlesNo, self._nVar))
        _particlePersonalBestObjective = np.ones(self._particlesNo) * val

        # Initialize the Particle global best
        _particleGlobalBestPosition = np.zeros(self._nVar)
        _particleGlobalBestObjective = val

        #endregion

        #region 3. [Report Parameters]

        _numberOfObjectiveFunctionCall = 0
        _particlesPersonalBestObjectiveIteration = np.ones((self._iterationCount, self._particlesNo)) * val
        _particlesPersonalBestPositionIteration = np.zeros((self._iterationCount, self._particlesNo, self._nVar))
        _particleGlobalBestObjectiveIteration = np.ones(self._iterationCount) * val
        _particleGlobalBestPositionIteration = np.ones((self._iterationCount, self._nVar))

        _particlesPositionIterAllParam = np.zeros((self._iterationCount, self._particlesNo, self._nVar))
        _weightHistory = np.empty(self._iterationCount)

        #endregion

        # region 4. Main Loop

        for iter in range(self._iterationCount):

            # region 4.1. First loop: Fitness calculation & updating particle bests (personal & global)
            # Positions & scores
            for i in range(self._particlesNo):  # For each particles
                # calculating the objective values
                fitness = self._selectedObjectiveFunction.functionCall(_particlesPosition[i, :], trainX ,trainY, testX, testY)
                _numberOfObjectiveFunctionCall += 1  # REPORT

                if self._toMinimize:
                    # Updating the best personal & global objectives and positions to minimize the problem
                    if fitness < _particlePersonalBestObjective[i]:
                        _particlePersonalBestObjective[i] = fitness
                        _particlePersonalBestPosition[i] = _particlesPosition[i, :]
                    if fitness < _particleGlobalBestObjective:
                        _particleGlobalBestObjective = fitness
                        _particleGlobalBestPosition = _particlesPosition[i, :]
                else:
                    # Updating the best personal & global objectives and positions to maximize the problem
                    if fitness > _particlePersonalBestObjective[i]:
                        _particlePersonalBestObjective[i] = fitness
                        _particlePersonalBestPosition[i] = _particlesPosition[i, :]
                    if fitness > _particleGlobalBestObjective:
                        _particleGlobalBestObjective = fitness
                        _particleGlobalBestPosition = _particlesPosition[i, :]

                # UPDATING THE REPORT VARIABLE --> Fitness history
                _particlesPositionIterAllParam[iter, i] = _particlesPosition[i, :]
                _particlesPersonalBestObjectiveIteration[iter, i] = _particlePersonalBestObjective[i]
                _particlesPersonalBestPositionIteration[iter, i] = _particlePersonalBestPosition[i]
                # End for-loop

            # endregion

            # region 4.2. Updating the Velocity & Position vectors

            # Calculating Velocity needs Weight to be calculated
            weight = self._weightMax - iter * (self._weightMax - self._weightMin) / self._iterationCount  # like a in GWO decreses linearly
            _weightHistory[iter] = weight

            for p in range(self._particlesNo):
                # Calculating Inertia
                inertia = weight * _particlesVelocity[p]
                # Calculating Cognitive Component
                r1 = GeneralFunctions.generateRandomUniform(elemNo=self._nVar)
                cognitiveComponent = c1 * r1 * (_particlePersonalBestPosition[p] - _particlesPosition[p])
                # Calculating Social Component
                r2 = GeneralFunctions.generateRandomUniform(elemNo=self._nVar)
                socialComponent = c2 * r2 * (_particleGlobalBestPosition - _particlesPosition[p])
                # Calculating Particles' Velocity
                _particlesVelocity[p] = inertia + cognitiveComponent + socialComponent

                # CHECK Velocity to be in range
                for i in range(self._nVar):
                    if _particlesVelocity[p][i] > velocityMax:
                        _particlesVelocity[p][i] = velocityMax
                    if _particlesVelocity[p][i] < velocityMin:
                        _particlesVelocity[p][i] = velocityMin

                # Updating the Particles' Position
                _particlesPosition[p] = _particlesPosition[p] + _particlesVelocity[p]

                # CHECK Positions to be in range
                for i in range(self._nVar):
                    if _particlesPosition[p][i] > ub:
                        _particlesPosition[p][i] = ub
                    if _particlesPosition[p][i] < lb:
                        _particlesPosition[p][i] = lb

            #endregion (4.2)

            # region 4.3. Save Results at the end of each Iteration
            _particleGlobalBestObjectiveIteration[iter] = _particleGlobalBestObjective
            _particleGlobalBestPositionIteration[iter] = _particleGlobalBestPosition
            _particlesPositionIterAllParam[iter] = _particlesPosition
            #endregion

            #region 4.4. SAVING RESULTS INTO a CLASS OBJECT
            psoResult = PSOresult(NumberOfParticles=self._particlesNo,
                                  NumberOfDimensions=self._nVar,
                                  NumberOfIterations=self._iterationCount,
                                  ObjectiveFunctionNumberOfCalls=_numberOfObjectiveFunctionCall,
                                  ObjectiveFunction = self._selectedObjectiveFunctionClass,
                                  GlobalBestObjectiveAllIterations=_particleGlobalBestObjectiveIteration,
                                  GlobalBestPositionAllIterations=_particleGlobalBestPositionIteration,
                                  PersonalBestObjectiveAllIterations=_particlesPersonalBestObjectiveIteration,
                                  PersonalBestPositionAllIterations=_particlesPersonalBestPositionIteration,
                                  PositionAllIterations=_particlesPositionIterAllParam,
                                  WeightHistory=_weightHistory)
            #endregion (4.4)

        #endregion (Main Loop)

        return psoResult

class PSO_FullyInformed():
    '''
    This Class is for Fully Informed PSO (Kennedy & Mendes 2003). The topology is All topology since we have the same things for the GOA
    '''
    def PSO(self, iterationCount, particlesNo, nVar, selectedObjectiveFunction, velocityCtrlCoeff,
            toMinimize, trainX=None, trainY=None, testX=None, testY=None,
            chi = 0.7298, cParam1 = 2.05, cParam2 = 2.05):

        #region 1. Reading inputs and set the attributes

        self._particlesNo = particlesNo  # The number of search agents
        self._nVar = nVar  # The number of parameters or dimensions
        self._iterationCount = iterationCount
        # self._weightMax = weightMax
        # self._weightMin = weightMin
        self._chi = chi
        self._velocityCtrlCoeff = velocityCtrlCoeff
        self._toMinimize = toMinimize
        self._selectedObjectiveFunctionClass = selectedObjectiveFunction
        self._selectedObjectiveFunction = selectedObjectiveFunction()

        # -------------- Reading required parameters from the selected function --------------
        bounds = self._selectedObjectiveFunction.functionBoundary
        lb = bounds[0]
        ub = bounds[1]
        lowerBound = lb * np.ones(self._nVar)
        upperBound = ub * np.ones(self._nVar)
        boundaryNo = upperBound.shape[0]  # OR lowerBound (No difference)

        c1 = cParam1
        c2 = cParam2
        velocityMax = (ub - lb) * self._velocityCtrlCoeff
        velocityMin = -velocityMax

        #endregion

        # region 2. [PSO-related Parameters]

        if self._toMinimize:
            val = np.inf
        else:
            val = -np.inf

        _particlesPosition = np.random.random((self._particlesNo, self._nVar)) * (upperBound - lowerBound) + lowerBound
        _particlesVelocity = np.zeros((self._particlesNo, self._nVar))
        _particleObjective = np.ones(self._particlesNo) * val
        _particlePersonalBestPosition = np.zeros((self._particlesNo, self._nVar))
        _particlePersonalBestObjective = np.ones(self._particlesNo) * val

        # Initialize the Particle global best
        _particleGlobalBestPosition = np.zeros(self._nVar)
        _particleGlobalBestObjective = val

        #endregion

        #region 3. [Report Parameters]

        _numberOfObjectiveFunctionCall = 0
        _particlesPersonalBestObjectiveIteration = np.ones((self._iterationCount, self._particlesNo)) * val
        _particlesPersonalBestPositionIteration = np.zeros((self._iterationCount, self._particlesNo, self._nVar))
        _particleGlobalBestObjectiveIteration = np.ones(self._iterationCount) * val
        _particleGlobalBestPositionIteration = np.ones((self._iterationCount, self._nVar))

        _particlesPositionIterAllParam = np.zeros((self._iterationCount, self._particlesNo, self._nVar))
        _particlesPositionIterAllCognitivePart = np.zeros((self._iterationCount, self._particlesNo, self._nVar))
        # _weightHistory = np.empty(self._iterationCount)

        #endregion

        # region 4. Main Loop

        for iter in range(self._iterationCount):

            # region 4.1. First loop: Fitness calculation & updating particle bests (personal & global)
            # Positions & scores
            for i in range(self._particlesNo):  # For each particles
                # calculating the objective values
                fitness = self._selectedObjectiveFunction.functionCall(_particlesPosition[i, :], trainX ,trainY, testX, testY)
                _numberOfObjectiveFunctionCall += 1  # REPORT

                if self._toMinimize:
                    # Updating the best personal & global objectives and positions to minimize the problem
                    if fitness < _particlePersonalBestObjective[i]:
                        _particlePersonalBestObjective[i] = fitness
                        _particlePersonalBestPosition[i] = _particlesPosition[i, :]
                    if fitness < _particleGlobalBestObjective:
                        _particleGlobalBestObjective = fitness
                        _particleGlobalBestPosition = _particlesPosition[i, :]
                else:
                    # Updating the best personal & global objectives and positions to maximize the problem
                    if fitness > _particlePersonalBestObjective[i]:
                        _particlePersonalBestObjective[i] = fitness
                        _particlePersonalBestPosition[i] = _particlesPosition[i, :]
                    if fitness > _particleGlobalBestObjective:
                        _particleGlobalBestObjective = fitness
                        _particleGlobalBestPosition = _particlesPosition[i, :]

                # UPDATING THE REPORT VARIABLE --> Fitness history
                _particlesPositionIterAllParam[iter, i] = _particlesPosition[i, :]
                _particlesPersonalBestObjectiveIteration[iter, i] = _particlePersonalBestObjective[i]
                _particlesPersonalBestPositionIteration[iter, i] = _particlePersonalBestPosition[i]
                # End for-loop

            # endregion

            # region 4.2. Updating the Velocity & Position vectors

            # # Calculating Velocity needs Weight to be calculated
            # weight = self._weightMax - iter * (self._weightMax - self._weightMin) / self._iterationCount  # like a in GWO decreses linearly
            # _weightHistory[iter] = weight

            for p in range(self._particlesNo):
                # Calculating Inertia
                velocityOld = _particlesVelocity[p]
                # Calculating Cognitive Component
                r = GeneralFunctions.generateRandomUniform(elemNo=self._nVar)
                gama = (c1 + c2) * r

                # Cognitive part
                vCurrentParticle = np.zeros(self._nVar)
                for i in range(self._particlesNo):
                    # Current particle = p
                    vCurrentParticle += _particlePersonalBestPosition[i] - _particlesPosition[p]
                _particlesPositionIterAllCognitivePart[iter,p] = vCurrentParticle

                cognitiveComponent = (gama * vCurrentParticle) / self._particlesNo

                # Calculating Particles' Velocity
                _particlesVelocity[p] = self._chi * (velocityOld + cognitiveComponent)

                # CHECK Velocity to be in range
                for i in range(self._nVar):
                    if _particlesVelocity[p][i] > velocityMax:
                        _particlesVelocity[p][i] = velocityMax
                    if _particlesVelocity[p][i] < velocityMin:
                        _particlesVelocity[p][i] = velocityMin

                # Updating the Particles' Position
                _particlesPosition[p] = _particlesPosition[p] + _particlesVelocity[p]

                # CHECK Positions to be in range
                for i in range(self._nVar):
                    if _particlesPosition[p][i] > ub:
                        _particlesPosition[p][i] = ub
                    if _particlesPosition[p][i] < lb:
                        _particlesPosition[p][i] = lb

            #endregion (4.2)

            # region 4.3. Save Results at the end of each Iteration
            _particleGlobalBestObjectiveIteration[iter] = _particleGlobalBestObjective
            _particleGlobalBestPositionIteration[iter] = _particleGlobalBestPosition
            _particlesPositionIterAllParam[iter] = _particlesPosition
            #endregion

            #region 4.4. SAVING RESULTS INTO a CLASS OBJECT
            psoResult = PSOresult(NumberOfParticles=self._particlesNo,
                                  NumberOfDimensions=self._nVar,
                                  NumberOfIterations=self._iterationCount,
                                  ObjectiveFunctionNumberOfCalls=_numberOfObjectiveFunctionCall,
                                  ObjectiveFunction = self._selectedObjectiveFunctionClass,
                                  GlobalBestObjectiveAllIterations=_particleGlobalBestObjectiveIteration,
                                  GlobalBestPositionAllIterations=_particleGlobalBestPositionIteration,
                                  PersonalBestObjectiveAllIterations=_particlesPersonalBestObjectiveIteration,
                                  PersonalBestPositionAllIterations=_particlesPersonalBestPositionIteration,
                                  PositionAllIterations=_particlesPositionIterAllParam,
                                  WeightHistory=np.nan)
            #endregion (4.4)

        #endregion (Main Loop)

        return psoResult

class PSO_BareBones():

    def PSO(self, iterationCount, particlesNo, nVar, selectedObjectiveFunction, toMinimize,
            trainX=None, trainY=None, testX=None, testY=None):

        #region 1. Reading inputs and set the attributes

        self._particlesNo = particlesNo  # The number of search agents
        self._nVar = nVar  # The number of parameters or dimensions
        self._iterationCount = iterationCount
        self._toMinimize = toMinimize
        self._selectedObjectiveFunctionClass = selectedObjectiveFunction
        self._selectedObjectiveFunction = selectedObjectiveFunction()

        # -------------- Reading required parameters from the selected function --------------
        bounds = self._selectedObjectiveFunction.functionBoundary
        lb = bounds[0]
        ub = bounds[1]
        lowerBound = lb * np.ones(self._nVar)
        upperBound = ub * np.ones(self._nVar)
        boundaryNo = upperBound.shape[0]  # OR lowerBound (No difference)

        #endregion

        # region 2. [PSO-related Parameters]

        if self._toMinimize:
            val = np.inf
        else:
            val = -np.inf

        _particlesPosition = np.random.random((self._particlesNo, self._nVar)) * (upperBound - lowerBound) + lowerBound
        _particlesVelocity = np.zeros((self._particlesNo, self._nVar))
        _particleObjective = np.ones(self._particlesNo) * val
        _particlePersonalBestPosition = np.zeros((self._particlesNo, self._nVar))
        _particlePersonalBestObjective = np.ones(self._particlesNo) * val

        # Initialize the Particle global best
        _particleGlobalBestPosition = np.zeros(self._nVar)
        _particleGlobalBestObjective = val

        #endregion

        #region 3. [Report Parameters]

        _numberOfObjectiveFunctionCall = 0
        _particlesPersonalBestObjectiveIteration = np.ones((self._iterationCount, self._particlesNo)) * val
        _particlesPersonalBestPositionIteration = np.zeros((self._iterationCount, self._particlesNo, self._nVar))
        _particleGlobalBestObjectiveIteration = np.ones(self._iterationCount) * val
        _particleGlobalBestPositionIteration = np.ones((self._iterationCount, self._nVar))

        _particlesPositionIterAllParam = np.zeros((self._iterationCount, self._particlesNo, self._nVar))

        #endregion

        # region 4. Main Loop

        for iter in range(self._iterationCount):

            # region 4.1. First loop: Fitness calculation & updating particle bests (personal & global)
            # Positions & scores
            for i in range(self._particlesNo):  # For each particles
                # calculating the objective values
                fitness = self._selectedObjectiveFunction.functionCall(_particlesPosition[i, :], trainX ,trainY, testX, testY)
                _numberOfObjectiveFunctionCall += 1  # REPORT

                if self._toMinimize:
                    # Updating the best personal & global objectives and positions to minimize the problem
                    if fitness < _particlePersonalBestObjective[i]:
                        _particlePersonalBestObjective[i] = fitness
                        _particlePersonalBestPosition[i] = _particlesPosition[i, :]
                    if fitness < _particleGlobalBestObjective:
                        _particleGlobalBestObjective = fitness
                        _particleGlobalBestPosition = _particlesPosition[i, :]
                else:
                    # Updating the best personal & global objectives and positions to maximize the problem
                    if fitness > _particlePersonalBestObjective[i]:
                        _particlePersonalBestObjective[i] = fitness
                        _particlePersonalBestPosition[i] = _particlesPosition[i, :]
                    if fitness > _particleGlobalBestObjective:
                        _particleGlobalBestObjective = fitness
                        _particleGlobalBestPosition = _particlesPosition[i, :]

                # UPDATING THE REPORT VARIABLE --> Fitness history
                _particlesPositionIterAllParam[iter, i] = _particlesPosition[i, :]
                _particlesPersonalBestObjectiveIteration[iter, i] = _particlePersonalBestObjective[i]
                _particlesPersonalBestPositionIteration[iter, i] = _particlePersonalBestPosition[i]
                # End for-loop

            # endregion

            # region 4.2. Updating the Position vector

            for p in range(self._particlesNo):
                # Calculating g
                g = 0.5 * (_particlePersonalBestPosition[p] + _particleGlobalBestPosition)
                sigma = np.absolute(_particleGlobalBestPosition - _particlePersonalBestPosition[p])

                # Calculating random Component
                rn = GeneralFunctions.generateRandomN(elemNo=self._nVar)

                # Updating the Particles' Position
                _particlesPosition[p] = g + rn * sigma

                # CHECK Positions to be in range
                for i in range(self._nVar):
                    if _particlesPosition[p][i] > ub:
                        _particlesPosition[p][i] = ub
                    if _particlesPosition[p][i] < lb:
                        _particlesPosition[p][i] = lb

            #endregion (4.2)

            # region 4.3. Save Results at the end of each Iteration
            _particleGlobalBestObjectiveIteration[iter] = _particleGlobalBestObjective
            _particleGlobalBestPositionIteration[iter] = _particleGlobalBestPosition
            _particlesPositionIterAllParam[iter] = _particlesPosition
            #endregion

            #region 4.4. SAVING RESULTS INTO a CLASS OBJECT

            psoResult = PSOresult(NumberOfParticles=self._particlesNo,
                                  NumberOfDimensions=self._nVar,
                                  NumberOfIterations=self._iterationCount,
                                  ObjectiveFunctionNumberOfCalls=_numberOfObjectiveFunctionCall,
                                  ObjectiveFunction = self._selectedObjectiveFunctionClass,
                                  GlobalBestObjectiveAllIterations=_particleGlobalBestObjectiveIteration,
                                  GlobalBestPositionAllIterations=_particleGlobalBestPositionIteration,
                                  PersonalBestObjectiveAllIterations=_particlesPersonalBestObjectiveIteration,
                                  PersonalBestPositionAllIterations=_particlesPersonalBestPositionIteration,
                                  PositionAllIterations=_particlesPositionIterAllParam,
                                  WeightHistory = np.nan)
            #endregion (4.4)

        #endregion (Main Loop)

        return psoResult

class PSO_BareBonesFullyInformed():

    def PSO(self, iterationCount, particlesNo, nVar, selectedObjectiveFunction, alpha, toMinimize,
            trainX=None, trainY=None, testX=None, testY=None):

        # region 1. Reading inputs and set the attributes

        self._particlesNo = particlesNo  # The number of search agents
        self._nVar = nVar  # The number of parameters or dimensions
        self._iterationCount = iterationCount
        self._alpha = alpha
        self._toMinimize = toMinimize
        self._selectedObjectiveFunctionClass = selectedObjectiveFunction
        self._selectedObjectiveFunction = selectedObjectiveFunction()

        # -------------- Reading required parameters from the selected function --------------
        bounds = self._selectedObjectiveFunction.functionBoundary
        lb = bounds[0]
        ub = bounds[1]
        lowerBound = lb * np.ones(self._nVar)
        upperBound = ub * np.ones(self._nVar)
        boundaryNo = upperBound.shape[0]  # OR lowerBound (No difference)

        # endregion

        # region 2. [PSO-related Parameters]

        if self._toMinimize:
            val = np.inf
        else:
            val = -np.inf

        _particlesPosition = np.random.random((self._particlesNo, self._nVar)) * (upperBound - lowerBound) + lowerBound
        _particlesVelocity = np.zeros((self._particlesNo, self._nVar))
        _particleObjective = np.ones(self._particlesNo) * val
        _particlePersonalBestPosition = np.zeros((self._particlesNo, self._nVar))
        _particlePersonalBestObjective = np.ones(self._particlesNo) * val

        # Initialize the Particle global best
        _particleGlobalBestPosition = np.zeros(self._nVar)
        _particleGlobalBestObjective = val

        # endregion

        # region 3. [Report Parameters]

        _numberOfObjectiveFunctionCall = 0
        _particlesPersonalBestObjectiveIteration = np.ones((self._iterationCount, self._particlesNo)) * val
        _particlesPersonalBestPositionIteration = np.zeros((self._iterationCount, self._particlesNo, self._nVar))
        _particleGlobalBestObjectiveIteration = np.ones(self._iterationCount) * val
        _particleGlobalBestPositionIteration = np.ones((self._iterationCount, self._nVar))

        _particlesPositionIterAllParam = np.zeros((self._iterationCount, self._particlesNo, self._nVar))

        # endregion

        # region 4. Main Loop

        for iter in range(self._iterationCount):

            # region 4.1. First loop: Fitness calculation & updating particle bests (personal & global)
            # Positions & scores
            for i in range(self._particlesNo):  # For each particles
                # calculating the objective values
                fitness = self._selectedObjectiveFunction.functionCall(_particlesPosition[i, :], trainX, trainY, testX,
                                                                       testY)
                _numberOfObjectiveFunctionCall += 1  # REPORT

                if self._toMinimize:
                    # Updating the best personal & global objectives and positions to minimize the problem
                    if fitness < _particlePersonalBestObjective[i]:
                        _particlePersonalBestObjective[i] = fitness
                        _particlePersonalBestPosition[i] = _particlesPosition[i, :]
                    if fitness < _particleGlobalBestObjective:
                        _particleGlobalBestObjective = fitness
                        _particleGlobalBestPosition = _particlesPosition[i, :]
                else:
                    # Updating the best personal & global objectives and positions to maximize the problem
                    if fitness > _particlePersonalBestObjective[i]:
                        _particlePersonalBestObjective[i] = fitness
                        _particlePersonalBestPosition[i] = _particlesPosition[i, :]
                    if fitness > _particleGlobalBestObjective:
                        _particleGlobalBestObjective = fitness
                        _particleGlobalBestPosition = _particlesPosition[i, :]

                # UPDATING THE REPORT VARIABLE --> Fitness history
                _particlesPositionIterAllParam[iter, i] = _particlesPosition[i, :]
                _particlesPersonalBestObjectiveIteration[iter, i] = _particlePersonalBestObjective[i]
                _particlesPersonalBestPositionIteration[iter, i] = _particlePersonalBestPosition[i]
                # End for-loop

            # endregion

            # region 4.2. Updating the Velocity & Position vectors

            g = np.sum(_particlePersonalBestPosition, axis=0) / self._particlesNo

            for p in range(self._particlesNo):

                sigma = np.absolute(_particleGlobalBestPosition - _particlePersonalBestPosition[p])

                # Calculating random Component
                rn = GeneralFunctions.generateRandomN(elemNo=self._nVar)

                # Updating the Particles' Position
                _particlesPosition[p] = g + (self._alpha * rn * sigma)

                # CHECK Positions to be in range
                for i in range(self._nVar):
                    if _particlesPosition[p][i] > ub:
                        _particlesPosition[p][i] = ub
                    if _particlesPosition[p][i] < lb:
                        _particlesPosition[p][i] = lb

            # endregion (4.2)

            # region 4.3. Save Results at the end of each Iteration
            _particleGlobalBestObjectiveIteration[iter] = _particleGlobalBestObjective
            _particleGlobalBestPositionIteration[iter] = _particleGlobalBestPosition
            _particlesPositionIterAllParam[iter] = _particlesPosition
            # endregion

            # region 4.4. SAVING RESULTS INTO a CLASS OBJECT
            psoResult = PSOresult(NumberOfParticles=self._particlesNo,
                                  NumberOfDimensions=self._nVar,
                                  NumberOfIterations=self._iterationCount,
                                  ObjectiveFunctionNumberOfCalls=_numberOfObjectiveFunctionCall,
                                  ObjectiveFunction=self._selectedObjectiveFunctionClass,
                                  GlobalBestObjectiveAllIterations=_particleGlobalBestObjectiveIteration,
                                  GlobalBestPositionAllIterations=_particleGlobalBestPositionIteration,
                                  PersonalBestObjectiveAllIterations=_particlesPersonalBestObjectiveIteration,
                                  PersonalBestPositionAllIterations=_particlesPersonalBestPositionIteration,
                                  PositionAllIterations=_particlesPositionIterAllParam,
                                  WeightHistory=np.nan)
            # endregion (4.4)

        # endregion (Main Loop)

        return psoResult

#endregion

#endregion (PSO)

#region ------------------- Grasshopper Optimization Algorithm (GOA) -------------------

#region Abstract Class and Drived Classes for S Function in GOA

class GOAsFunction(ABC):

    @staticmethod
    @abstractmethod
    def functionCall(distance, intensityOfAttraction = 0.5, attractiveLengthScale = 1.5):
        pass

class GOAsStandard(GOAsFunction):

    def functionCall(distance, intensityOfAttraction = 0.5, attractiveLengthScale = 1.5):
        """Grasshopper Optimisation Algorithm: Theory and application - Eq.(2.3)"""
        # intensityOfAttraction = f
        # attractiveLengthScale = l
        return (intensityOfAttraction * np.exp(-distance / attractiveLengthScale) - np.exp(-distance))

#endregion

#region GOA Results

@dataclass
class GOAresult:
    """Class for returning result of GOA algorithm"""
    _NumberOfGrasshoppers: int
    _NumberOfDimensions: int
    _NumberOfIterations: int

    _ObjectiveFunctionNumberOfCalls: int
    _ObjectiveFunction: cls_ObjectiveFunctions.ObjectiveFunction
    _sFunction: GOAsFunction

    _GrasshopperBestObjectiveAllIterations: np.array
    _GrasshopperBestPositionAllIterations: np.array
    _GrasshopperObjectiveAllIterations: np.array
    _GrasshopperPositionAllIterations: np.array

    _GrasshopperSiAllIterations: np.array

    _cHistory: np.array

    # region [FUNCTIONS] INIT()
    def __init__(self, NumberOfGrasshoppers: int, NumberOfDimensions: int, NumberOfIterations: int,
                 ObjectiveFunctionNumberOfCalls: int, ObjectiveFunction: cls_ObjectiveFunctions.ObjectiveFunction,
                 sFunction: GOAsFunction, GrasshopperBestObjectiveAllIterations: np.array,
                 GrasshopperBestPositionAllIterations: np.array, GrasshopperObjectiveAllIterations: np.array,
                 GrasshopperPositionAllIterations: np.array, cHistory: np.array,
                 GrasshopperSiAllIterations: np.array,
                 GrasshopperDistAllIterations: np.array):

        self._ObjectiveFunction = ObjectiveFunction
        self._NumberOfGrasshoppers = NumberOfGrasshoppers
        self._NumberOfDimensions = NumberOfDimensions
        self._NumberOfIterations = NumberOfIterations
        self._ObjectiveFunctionNumberOfCalls = ObjectiveFunctionNumberOfCalls
        self._GrasshopperBestObjectiveAllIterations = GrasshopperBestObjectiveAllIterations
        self._GrasshopperBestPositionAllIterations = GrasshopperBestPositionAllIterations
        self._GrasshopperObjectiveAllIterations = GrasshopperObjectiveAllIterations
        self._GrasshopperPositionAllIterations = GrasshopperPositionAllIterations
        self._cHistory = cHistory
        self._sFunction = sFunction

        self._GrasshopperSiAllIterations = GrasshopperSiAllIterations
        self._GrasshopperDistAllIterations = GrasshopperDistAllIterations

    # endregion

    # region [FUNCTIONS] PRINT RESULT
    def printData(self):
        str = f"Number of Grasshoppers: {self._NumberOfGrasshoppers}\nNumber of Dimensions: {self._NumberOfDimensions} \nNumber of Iterations: {self._NumberOfIterations}\n" \
              f"Objective Function: {self._ObjectiveFunction}\nNumber of Function Evaluations: {self._ObjectiveFunctionNumberOfCalls}\n" \
              f"Global best objective: {self._GrasshopperBestObjectiveAllIterations[-1]}\n" \
              f"Global best position: {self._GrasshopperBestPositionAllIterations[-1]}"

        return str

    # endregion

    def plotAll_Plotly(self, plotTitle, is2D = True, show = True, save = False, savePath = ""):
        func = self._ObjectiveFunction()
        xOptimal = func.optimalX
        funcBounds = func.functionBoundary
        funcBoundsL = funcBounds[0]
        funcBoundsU = funcBounds[1]

        fig = make_subplots(rows=1,
                            cols=6,
                            x_title='',
                            y_title=' ',
                            subplot_titles=("Search History", "Trajectory of Grasshoppers 1st Dimension",
                                            "Trajectory of Grasshoppers (PCA)", "Average fitness of Grasshoppers",
                                            "Average, Min, MaxFitness of Grasshoppers", "Convergence Plot"))

        x1 = np.arange(1, self._NumberOfIterations + 1)
        # 1) Search History
        # -----------------------------------------------------------------------------------------
        # Plotly Version
        x = np.linspace(funcBoundsL, funcBoundsU, 500)
        y = np.linspace(funcBoundsL, funcBoundsU, 500)
        X, Y = np.meshgrid(x, y)

        XY = np.array([X, Y])
        Z = np.apply_along_axis(func.functionCall, 0, XY)

        yOptimal = func.optimalF

        if is2D:
            position = self._GrasshopperPositionAllIterations
            for i in range(self._NumberOfIterations):
                fig.add_scatter(x=position[i, :, 0], y=position[i, :, 1], mode="markers", marker_color="black",
                                marker_size=7, opacity=0.7, row=1, col=1)

        # Adding Contour Plot
        fig.add_trace(go.Contour(z=Z, x=x, y=y, contours_coloring='lines', line_width=2, colorscale="Viridis"), row=1,
                      col=1)
        fig.update_coloraxes(showscale=False)

        # Adding Global Optimum
        fig.add_scatter(x=[xOptimal], y=[xOptimal], mode="markers", marker_color="red", marker_symbol="x",
                        marker_size=20, row=1, col=1)

        # 2) Trajectory of each Grasshoppers in First Dimension
        # ------------------------------------------------------------------------------------------
        fig.add_trace(go.Scatter(x=x1,
                                 y=self._GrasshopperPositionAllIterations[:, 0, 0],
                                 mode='lines', name=f"Grasshopper {1}"), row=1, col=2)
        for i in range(1, self._NumberOfGrasshoppers):
            fig.add_trace(go.Scatter(x=x1, y=self._GrasshopperPositionAllIterations[:, i, 0],
                                     mode="lines", name=f"Grasshopper {i + 1}"), row=1, col=2)

        fig.update_xaxes(title_text=f"Iteration", row=1, col=2)
        fig.update_yaxes(title_text=f"1st dimension of each g. position", row=1, col=2)

        # 3) Trajectory of each Grasshoppers (PCA))
        # ------------------------------------------------------------------------------------------
        fig.add_trace(go.Scatter(x=x1,
                                 y=self.GetPositionsHistoryReducedDimension(0, 2)[:, 0],
                                 mode='lines', name=f"Grasshopper {1}"), row=1, col=3)
        for i in range(1, self._NumberOfGrasshoppers):
            fig.add_trace(go.Scatter(x=x1, y=self.GetPositionsHistoryReducedDimension(i, 2)[:, 0],
                                     mode="lines", name=f"Grasshopper {i + 1}"), row=1, col=3)

        fig.update_xaxes(title_text=f"Iteration", row=1, col=3)
        fig.update_yaxes(title_text=f"PCA of each grasshopper position", row=1, col=3)

        # 4) Average Fitness of Grasshoppers
        # ------------------------------------------------------------------------------------------
        fig.add_trace(go.Scatter(x=x1,
                                 y=np.average(self._GrasshopperObjectiveAllIterations, axis=1),
                                 mode='lines', name=f"Average Objective value"), row=1, col=4)

        fig.add_hline(y=yOptimal, line_width=3, line_dash="dash", line_color="red",
                      annotation_text="Optimal value", annotation_position="bottom right",
                      row=1, col=4)

        fig.update_xaxes(title_text=f"Iteration", row=1, col=4)
        fig.update_yaxes(title_text=f"Fitness", row=1, col=4)

        # 5) Average, Min, and Max Fitness of Grasshoppers
        # ------------------------------------------------------------------------------------------
        fig.add_trace(go.Scatter(x=x1,
                                 y=np.mean(self._GrasshopperObjectiveAllIterations, axis=1),
                                 mode='lines', name=f"Average Objective value"), row=1, col=5)

        fig.add_trace(go.Scatter(x=x1,
                                 y=np.min(self._GrasshopperObjectiveAllIterations, axis=1),
                                 mode='lines', name=f"Minimum Objective value"), row=1, col=5)

        fig.add_trace(go.Scatter(x=x1,
                                 y=np.max(self._GrasshopperObjectiveAllIterations, axis=1),
                                 mode='lines', name=f"Maximum Objective value"), row=1, col=5)

        fig.add_hline(y=yOptimal, line_width=3, line_dash="dash", line_color="red",
                      annotation_text="Optimal value", annotation_position="bottom right",
                      row=1, col=5)

        fig.update_xaxes(title_text=f"Iteration", row=1, col=5)
        fig.update_yaxes(title_text=f"Fitness", row=1, col=5)

        # 6) Convergence Plot
        # ------------------------------------------------------------------------------------------

        x1 = np.arange(1, self._NumberOfIterations + 1)
        fig.add_trace(go.Scatter(x=x1,
                                 y=self._GrasshopperBestObjectiveAllIterations,
                                 mode='lines', name=f"Fitness"), row=1, col=6)

        fig.add_hline(y=yOptimal, line_width=3, line_dash="dash", line_color="red",
                      annotation_text="Optimal value", annotation_position="bottom right",
                      row=1, col=6)

        fig.update_xaxes(title_text=f"Iteration", row=1, col=6)
        fig.update_yaxes(title_text=f"Fitness", row=1, col=6)

        # ------------------------------------------------------------------------------------------

        fig.update_layout(showlegend=False,
                          title = plotTitle,
                          title_x=0.5,
                          font_family="Courier New",
                          font_color="black",
                          font_size = 12,
                          title_font_family="Times New Roman",
                          title_font_color="red",
                          title_font_size = 25,
                          legend_title_font_color="green",
                          width=2200
                          )
        if save:
            fig.write_image(savePath)
        if show:
            fig.show()


    # region [FUNCTIONS]
    def GetBestPositionsReducedDimension(self, toDimension):
        return GeneralFunctions.getReducedDimensionByPCA(self._GrasshopperBestPositionAllIterations, toDimension)

    def GetPositionsHistoryReducedDimension(self, grasshopperNum, toDimension):
        return GeneralFunctions.getReducedDimensionByPCA(self._GrasshopperPositionAllIterations[:,grasshopperNum], toDimension)

    def GetPositionsHistoryOneIterationReducedDimension(self, iteration, toDimension):
        return GeneralFunctions.getReducedDimensionByPCA(self._GrasshopperPositionAllIterations[iteration,:], toDimension)

    # endregion

    # region ATTRIBUTES [SET & GET]

    def _get_NumberOfGrasshoppers(self):
        return self._NumberOfGrasshoppers
    def _set_NumberOfGrasshoppers(self, value):
        self._NumberOfGrasshoppers = value
    numberOfGrasshoppers = property(fget=_get_NumberOfGrasshoppers, fset=_set_NumberOfGrasshoppers, doc="The Number of Grasshoppers")

    def _get_NumberOfDimensions(self):
        return self._NumberOfDimensions
    def _set_NumberOfDimensions(self, value):
        self._NumberOfDimensions = value
    numberOfDimensions = property(fget=_get_NumberOfDimensions, fset=_set_NumberOfDimensions, doc="The Number of Dimensions")

    def _get_NumberOfIterations(self):
        return self._NumberOfIterations
    def _set_NumberOfIterations(self, value):
        self._NumberOfIterations = value
    numberOfIterations = property(fget=_get_NumberOfIterations, fset=_set_NumberOfIterations, doc="The Number of Iterations")

    def _get_ObjectiveFunction(self):
        return self._ObjectiveFunction
    def _set_ObjectiveFunction(self, value):
        self._ObjectiveFunction = value
    objectiveFunction = property(fget=_get_ObjectiveFunction, fset=_set_ObjectiveFunction, doc="The Objective Function")

    def _get_ObjectiveFunctionNumberOfCalls(self):
        return self._ObjectiveFunctionNumberOfCalls
    objectiveFunctionNumberOfCalls = property(fget=_get_ObjectiveFunctionNumberOfCalls,
                                              doc="The number of Objective Function Evaluations")

    def _get_GrasshopperBestObjectiveAllIterations(self):
        return self._GrasshopperBestObjectiveAllIterations
    grasshopperBestObjectiveAllIteration = property(fget=_get_GrasshopperBestObjectiveAllIterations,
                                               doc="Global best fitness array in all iterations")

    def _get_GrasshopperBestPositionAllIterations(self):
        return self._GrasshopperBestPositionAllIterations
    grasshopperBestPositionAllIteration = property(fget=_get_GrasshopperBestPositionAllIterations,
                                              doc="Global best position array in all iterations")

    def _get_GrasshopperObjectiveAllIterations(self):
        return self._GrasshopperObjectiveAllIterations
    grasshopperObjectiveAllIteration = property(fget=_get_GrasshopperObjectiveAllIterations,
                                                 doc="The fitness array in all iterations")

    def _get_grasshopperPositionAllIterations(self):
        return self._GrasshopperPositionAllIterations
    grasshopperPositionAllIteration = property(fget=_get_grasshopperPositionAllIterations,
                                                doc="The position array in all iterations")

    def _get_cHistory(self):
        return self._cHistory
    cHistory = property(fget=_get_cHistory, doc="c array in all iterations")

    def _get_SiHistory(self):
        return self._GrasshopperSiAllIterations
    SiHistory = property(fget=_get_SiHistory, doc="Si array in all iterations for each grasshopper")

    def _get_DistHistory(self):
        return self._GrasshopperDistAllIterations
    DistHistory = property(fget=_get_DistHistory, doc="Dist array in all iterations for each grasshopper and its distance to other grasshoppers")

    # endregion

#endregion (GOA Results)

#region MAIN GOA

class GOA():

    def _calculateDistance(self, a, b):
        return GeneralFunctions.getEuclideanDistanceBetweenTwoPoint(a,b)

    def _calculateSocialInteraction(self, grasshopperPositions, currentGrasshopperIndex, grasshoppersNo, nVar, lb, ub,
                                    cParam, selectedSFunction, currentIteration):
        if (self._mealpyGOA):
            EPSILON = 10E-10 # Based on Mealpy library
        else:
            EPSILON = np.finfo(float).eps
        S_i_total = np.zeros(nVar)
        # Distance between current grasshopper and i^th grasshopper
        for i in range(grasshoppersNo):
            dist = self._calculateDistance(grasshopperPositions[currentGrasshopperIndex], grasshopperPositions[i])
            self._DistAllIteration[currentIteration,currentGrasshopperIndex, i] = dist

            # xj - xi / dij in Eq.(2.7)
            distUnitVector = (grasshopperPositions[currentGrasshopperIndex] - grasshopperPositions[i]) / (dist + EPSILON)
            # |xjd - xid| in Eq. (2.7)
            xj_xi = 2 + np.remainder(dist, 2)
            s = selectedSFunction.functionCall(xj_xi)
            cTimesBound = (cParam / 2) * (ub - lb)
            # Eq.(2.7)
            S_i_total += (cTimesBound * s * distUnitVector)

        return S_i_total

    def _GOAInitialization(self, lowerBound, upperBound, boundaryNo, grasshopperNo, nVar):
        """Initializing position of the grasshopperNo"""
        if boundaryNo == 1:  # one boundary
            positions = np.random.random((grasshopperNo, nVar)) * (upperBound - lowerBound) + lowerBound
        else:  # more than one boundary
            positions = np.zeros((grasshopperNo, nVar))
            for i in range(grasshopperNo):  # For each grasshopper
                for j in range(nVar):  # For each variable (dimension)
                    upperBound_j = upperBound[j]
                    lowerBound_j = lowerBound[j]
                    positions[i, j] = np.random.uniform() * (upperBound_j - lowerBound_j) + lowerBound_j

        return positions

    '''The original algorithm should be run with a even number of variables!'''
    def GOA(self, iterationCount, grasshoppersNo, nVar, selectedObjectiveFunction, selectedsFunction, toMinimize,
            trainX=None, trainY=None, testX=None, testY=None, mealpyGOA = False, cMin = 0.00004, cMax = 1.0):

        # region 1. Reading inputs and set the attributes

        self._grasshoppersNo = grasshoppersNo  # The number of search agents
        # This algorithm should be run with a even number of variables!
        if nVar % 2 != 0:
            nVar = nVar + 1
        self._nVar = nVar  # The number of parameters or dimensions
        self._iterationCount = iterationCount
        self._toMinimize = toMinimize
        self._mealpyGOA = mealpyGOA
        self._selectedObjectiveFunctionClass = selectedObjectiveFunction
        self._selectedObjectiveFunction = selectedObjectiveFunction()
        self._selectedsFunction = selectedsFunction
        self._DistAllIteration = np.empty((self._iterationCount, self._grasshoppersNo, self._grasshoppersNo))

        # -------------- Reading required parameters from the selected function --------------
        bounds = self._selectedObjectiveFunction.functionBoundary
        lb = bounds[0]
        ub = bounds[1]
        lowerBound = lb * np.ones(self._nVar)
        upperBound = ub * np.ones(self._nVar)
        boundaryNo = upperBound.shape[0]  # OR lowerBound (No difference)

        # endregion

        # region 2. [GOA-related Parameters]

        _cMax = cMax
        _cMin = cMin

        if self._toMinimize:
            val = np.inf
        else:
            val = -np.inf

        _grasshopperPosition = self._GOAInitialization(lowerBound, upperBound, boundaryNo, self._grasshoppersNo, self._nVar)
        _grasshopperFitness = np.ones(self._grasshoppersNo) * val
        # GlobalBest = Target
        _grasshopperGlobalBestFitness = val
        _grasshopperGlobalBestPosition = np.empty(self._nVar)

        # endregion

        # region 3. [Report Parameters]

        _numberOfObjectiveFunctionCall = 0
        _cHistory = np.empty(self._iterationCount)
        _positionHistory = np.empty((self._iterationCount, self._grasshoppersNo, self._nVar))
        _fitnessHistory = np.empty((self._iterationCount, self._grasshoppersNo))
        _bestFitnessIteration = np.zeros(self._iterationCount) # convergence_curve
        _bestPositionIteration = np.empty((self._iterationCount, self._nVar))
        _siIteration = np.empty((self._iterationCount, self._grasshoppersNo, self._nVar))

        # endregion

        # region 4.1. First loop (Iteration =1)
        iter = 0
        # Fitness calculation of initial grasshoppers
        for i in range(self._grasshoppersNo):  # For each grasshopper
            # calculating the objective values
            _grasshopperFitness[i] = self._selectedObjectiveFunction.functionCall(_grasshopperPosition[i, :])
            # REPORT
            _numberOfObjectiveFunctionCall += 1
            _positionHistory[iter, i] = _grasshopperPosition[i, :]
            _fitnessHistory[iter, i] = _grasshopperFitness[i]

        # Finding the best grasshopper index & saving the fitness and position as target
        if self._toMinimize:
            bestIndex = np.argmin(_grasshopperFitness)
        else:
            bestIndex = np.argmax(_grasshopperFitness)

        _bestFitnessIteration[iter] = _grasshopperGlobalBestFitness = _grasshopperFitness[bestIndex]
        _bestPositionIteration[iter] = _grasshopperGlobalBestPosition = _grasshopperPosition[bestIndex,:]

        #endregion (4.1)

        #region 4.2. Main Loop
        _cHistory[iter] = _cMax
        iter += 1
        while iter < self._iterationCount:

            # Calculation c parameter - Eq.(2.8)
            _c = _cHistory[iter] = _cMax - iter * ((_cMax - _cMin)/ self._iterationCount)

            for i in range(self._grasshoppersNo):

                # Calculate SI (Social Interaction)
                si = self._calculateSocialInteraction(grasshopperPositions = _grasshopperPosition, currentGrasshopperIndex=i,
                                                      grasshoppersNo=self._grasshoppersNo, nVar=self._nVar, lb=lb, ub=ub,
                                                      cParam=_c, selectedSFunction=self._selectedsFunction,
                                                      currentIteration = iter)
                _siIteration[iter,i] = si

                # Updating the position for i_th grasshopper - Eq. (2.7) in the paper
                if self._mealpyGOA:
                    # Mealpy lib
                    rand = GeneralFunctions.generateRandomN(start=0, end=1, elemNo= self._nVar)
                else:
                    rand = 1
                _grasshopperPosition[i] = _c * rand * si + _grasshopperGlobalBestPosition

                # Checking the boundaries
                _grasshopperPosition[i] = np.minimum(np.maximum(_grasshopperPosition[i], lowerBound),upperBound)

                # Calculating new cost
                _grasshopperFitness[i] = self._selectedObjectiveFunction.functionCall(_grasshopperPosition[i, :])

                # REPORT
                _numberOfObjectiveFunctionCall += 1

            # Updating the global best fitness & position
            if self._toMinimize:
                bestIndex = np.argmin(_grasshopperFitness)
            else:
                bestIndex = np.argmax(_grasshopperFitness)

            # Updating global best and report variables
            _bestFitnessIteration[iter] = _grasshopperGlobalBestFitness = _grasshopperFitness[bestIndex]
            _bestPositionIteration[iter] = _grasshopperGlobalBestPosition = _grasshopperPosition[bestIndex, :]
            _positionHistory[iter] = _grasshopperPosition
            _fitnessHistory[iter] = _grasshopperFitness

            # Increasing loop variable
            iter += 1

        #endregion (END WHILE)

        # region 4.3. SAVING RESULTS INTO a CLASS OBJECT
        goaResult = GOAresult(NumberOfGrasshoppers=self._grasshoppersNo,
                              NumberOfDimensions=self._nVar,
                              NumberOfIterations=self._iterationCount,
                              ObjectiveFunctionNumberOfCalls=_numberOfObjectiveFunctionCall,
                              ObjectiveFunction=self._selectedObjectiveFunctionClass,
                              sFunction= self._selectedsFunction,
                              GrasshopperBestObjectiveAllIterations=_bestFitnessIteration,
                              GrasshopperBestPositionAllIterations=_bestPositionIteration,
                              GrasshopperObjectiveAllIterations=_fitnessHistory,
                              GrasshopperPositionAllIterations=_positionHistory,
                              cHistory=_cHistory,
                              GrasshopperSiAllIterations=_siIteration,
                              GrasshopperDistAllIterations = self._DistAllIteration)
        # endregion (4.4)

        return goaResult

#endregion (Main GOA)

# endregion
