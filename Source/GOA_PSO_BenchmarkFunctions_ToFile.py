# Import Libraries
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os

# My Libraries
import cls_SwarmIntelligence
from cls_SwarmIntelligence import PSO
from cls_SwarmIntelligence import PSO_FullyInformed
from cls_SwarmIntelligence import PSO_BareBones
from cls_SwarmIntelligence import PSO_BareBonesFullyInformed
from cls_SwarmIntelligence import GOA
import cls_ObjectiveFunctions as objectiveFunc

# 18 Benchmark Functions
benchmarkFuncList = [
    objectiveFunc.Ackley,
    objectiveFunc.Griewank,
    objectiveFunc.Levy,
    objectiveFunc.Rastrigin,
    objectiveFunc.Schwefel,
    objectiveFunc.Schwefel220,
    objectiveFunc.Schwefel221,
    objectiveFunc.Schwefel222,
    objectiveFunc.Schwefel223,
    objectiveFunc.Schwefel226,
    objectiveFunc.Quartic,
    objectiveFunc.RotatedHyperEllipsoid,
    objectiveFunc.Sphere,
    objectiveFunc.Step,
    objectiveFunc.SumSquares,
    objectiveFunc.Rosenbrock,
    objectiveFunc.GeneralizedPenalizedFunc1,
    objectiveFunc.GeneralizedPenalizedFunc2
    ]

# Setting for Swarm Algorithms
NumOfIterations = 150
NumOfIndividuals = 100
NumOfDimensions = 30
NumberOfRun = 200
summary = f"{NumOfIterations}Iter_{NumOfIndividuals}Indiv_{NumOfDimensions}Dim_{NumberOfRun}Run"
alpha = 1
# Setting for Random Numbers
randomSeed = 720
np.random.seed(randomSeed)

# **********************************************************************************************************************
lst = []
pos = 0
for benchmarkFunc in benchmarkFuncList:
    print("--------------------------------------------------------------")
    print("Processing " + benchmarkFunc().functionName + " Function ...")
    print("--------------------------------------------------------------")

    #region ================================================ GOA =======================================================

    print("Running GOA ...")
    rGOA_bestFitness = np.empty((NumberOfRun))
    goa = GOA()
    for r in range(NumberOfRun):
        startT = time.time()
        rGOA_multi = goa.GOA(grasshoppersNo=NumOfIndividuals, nVar=NumOfDimensions,
                                      iterationCount=NumOfIterations,
                                      selectedObjectiveFunction=benchmarkFunc,
                                      selectedsFunction=cls_SwarmIntelligence.GOAsStandard, cMin=0.00004, cMax=1.0,
                                      toMinimize=True)
        endT = time.time()
        elapsedTime = endT - startT

        tempLst = [pos,
                   rGOA_multi.grasshopperBestObjectiveAllIteration[-1],
                   rGOA_multi.grasshopperBestPositionAllIteration[-1],
                   rGOA_multi.grasshopperBestObjectiveAllIteration,
                   rGOA_multi.grasshopperBestPositionAllIteration,
                   rGOA_multi.grasshopperPositionAllIteration,
                   # rGOA_multi.DistHistory,
                   rGOA_multi.numberOfGrasshoppers,
                   rGOA_multi.numberOfDimensions,
                   rGOA_multi.numberOfIterations,
                   rGOA_multi.objectiveFunctionNumberOfCalls,
                   elapsedTime,
                   benchmarkFunc().functionName,
                   "GOA"]
        tempLstFinal = tuple(tempLst)
        lst.insert(pos, tempLstFinal)

        pos += 1

    #endregion

    #region =============================================== PSO ========================================================

    print("Running PSO ...")
    rPSO_bestFitness = np.empty((NumberOfRun))
    pso = PSO()
    for r in range(NumberOfRun):
        startT = time.time()
        rPSO_multi = pso.PSO(particlesNo=NumOfIndividuals, nVar=NumOfDimensions,
                                      iterationCount=NumOfIterations,
                                      selectedObjectiveFunction=benchmarkFunc, toMinimize=True,
                                      weightMax=0.9, weightMin=0.2, velocityCtrlCoeff=0.2,
                                      cParam1=2, cParam2=2)
        endT = time.time()
        elapsedTime = endT - startT

        tempLst = [pos,
                   rPSO_multi.globalBestObjectiveAllIteration[-1],
                   rPSO_multi.globalBestPositionAllIteration[-1],
                   rPSO_multi.globalBestObjectiveAllIteration,
                   rPSO_multi.globalBestPositionAllIteration,
                   rPSO_multi.positionsAllIteration,
                   rPSO_multi.numberOfParticles,
                   rPSO_multi.numberOfDimensions,
                   rPSO_multi.numberOfIterations,
                   rPSO_multi.objectiveFunctionNumberOfCalls,
                   elapsedTime,
                   benchmarkFunc().functionName,
                   "PSO"]
        tempLstFinal = tuple(tempLst)
        lst.insert(pos, tempLstFinal)
        pos += 1

    #endregion

    #region ================================================ FiPSO =====================================================

    print("Running FiPSO ...")
    rFiPSO_bestFitness= np.empty((NumberOfRun))
    pso = PSO_FullyInformed()
    for r in range(NumberOfRun):
        startT = time.time()
        rFiPSO_multi = pso.PSO(particlesNo=NumOfIndividuals, nVar=NumOfDimensions,
                                      iterationCount=NumOfIterations, velocityCtrlCoeff=0.2, chi = 0.7298,
                                      selectedObjectiveFunction=benchmarkFunc, toMinimize=True)
        endT = time.time()
        elapsedTime = endT - startT

        tempLst = [pos,
                   rFiPSO_multi.globalBestObjectiveAllIteration[-1],
                   rFiPSO_multi.globalBestPositionAllIteration[-1],
                   rFiPSO_multi.globalBestObjectiveAllIteration,
                   rFiPSO_multi.globalBestPositionAllIteration,
                   rFiPSO_multi.positionsAllIteration,
                   rFiPSO_multi.numberOfParticles,
                   rFiPSO_multi.numberOfDimensions,
                   rFiPSO_multi.numberOfIterations,
                   rFiPSO_multi.objectiveFunctionNumberOfCalls,
                   elapsedTime,
                   benchmarkFunc().functionName,
                   "FiPSO"]
        tempLstFinal = tuple(tempLst)
        lst.insert(pos, tempLstFinal)
        pos += 1

    #endregion

    #region ============================================= Bare-Bones PSO ===============================================

    print("Running Bare-Bones PSO ...")
    rBBPSO_bestFitness= np.empty((NumberOfRun))
    pso = PSO_BareBones()
    for r in range(NumberOfRun):
        startT = time.time()
        rBBPSO_multi = pso.PSO(particlesNo=NumOfIndividuals, nVar=NumOfDimensions,
                                      iterationCount=NumOfIterations,
                                      selectedObjectiveFunction=benchmarkFunc, toMinimize=True)
        endT = time.time()
        elapsedTime = endT - startT

        tempLst = [pos,
                   rBBPSO_multi.globalBestObjectiveAllIteration[-1],
                   rBBPSO_multi.globalBestPositionAllIteration[-1],
                   rBBPSO_multi.globalBestObjectiveAllIteration,
                   rBBPSO_multi.globalBestPositionAllIteration,
                   rBBPSO_multi.positionsAllIteration,
                   rBBPSO_multi.numberOfParticles,
                   rBBPSO_multi.numberOfDimensions,
                   rBBPSO_multi.numberOfIterations,
                   rBBPSO_multi.objectiveFunctionNumberOfCalls,
                   elapsedTime,
                   benchmarkFunc().functionName,
                   "BBPSO"]
        tempLstFinal = tuple(tempLst)
        lst.insert(pos, tempLstFinal)
        pos += 1

    #endregion

    #region ============================================ Bare-Bones FiPSO ==============================================

    print("Running Bare-Bones FiPSO ...")
    rBBFiPSO_bestFitness = np.empty((NumberOfRun))
    pso = PSO_BareBonesFullyInformed()
    for r in range(NumberOfRun):
        startT = time.time()
        rBBFiPSO_multi = pso.PSO(particlesNo=NumOfIndividuals, nVar=NumOfDimensions,
                                      iterationCount=NumOfIterations,
                                      selectedObjectiveFunction=benchmarkFunc, toMinimize=True,
                                      alpha=alpha)
        endT = time.time()
        elapsedTime = endT - startT

        tempLst = [pos,
                   rBBFiPSO_multi.globalBestObjectiveAllIteration[-1],
                   rBBFiPSO_multi.globalBestPositionAllIteration[-1],
                   rBBFiPSO_multi.globalBestObjectiveAllIteration,
                   rBBFiPSO_multi.globalBestPositionAllIteration,
                   rBBFiPSO_multi.positionsAllIteration,
                   rBBFiPSO_multi.numberOfParticles,
                   rBBFiPSO_multi.numberOfDimensions,
                   rBBFiPSO_multi.numberOfIterations,
                   rBBFiPSO_multi.objectiveFunctionNumberOfCalls,
                   elapsedTime,
                   benchmarkFunc().functionName,
                   "BBFiPSO"]
        tempLstFinal = tuple(tempLst)
        lst.insert(pos, tempLstFinal)
        pos += 1

    #endregion

# ============================================== Saving ALL Results ====================================================
# Main DataFrame
print("--------------------------------------------------------------")
print("Creating a DataFrame to Save Data ...")
dfResult = pd.DataFrame(lst, columns=["IDX", "BestObjective", "BestPosition", "BestObjectiveAllIteration", "BestPositionAllIteration",
                                      "PositionAllIteration","NumOfIndividuals", "NumOfDimensions", "NumOfIterations",
                                      "NumOfFunctionCall", "ElapsedTime", "ObjectiveFunction", "Algorithm"])

print("Creating a New Directory ...")
currentTime = "{date:%Y-%m-%d_%H-%M}".format(date=datetime.now())
# Create a Directory for Final Results
currentDir = "GOA_PSO_AllBenchmarkFuncs"
NewDir = os.path.join(currentDir, currentTime)

try:
    os.mkdir(NewDir)
except OSError as error:
    print(error)

print("Saving Numpy Arrays ...")
for index, row in dfResult.iterrows():
    IDX = row['IDX']
    BestPos = row["BestPosition"]
    BestPosAll = row["BestPositionAllIteration"]
    BestObjAll = row["BestObjectiveAllIteration"]
    PosAll = row["PositionAllIteration"]

    filePath = f"{NewDir}/{IDX}_NDArrayResult.npz"

    # Saving NDArrays for each entry into file to retrieve later
    np.savez(filePath,
             BestPosition = BestPos,
             BestPositionAllIteration = BestPosAll,
             BestObjectiveAllIteration = BestObjAll,
             PositionAllIteration = PosAll
             )

# REMOVING NDArray columns from DataFrame
print("Saving Data to CSV File ...")
dfResult.drop(["BestPosition", "BestObjectiveAllIteration", "BestPositionAllIteration", "PositionAllIteration"], axis=1, inplace=True)
dfResult.to_csv(f"{NewDir}/Results_{currentTime}.csv", index=False)

print("Saving Files Have Been Successfully Done!")


