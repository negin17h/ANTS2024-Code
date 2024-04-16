import os
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Visualization (Plotly)
import matplotlib.pyplot as plt
import seaborn as sns

import cls_ObjectiveFunctions as objectiveFunc

# 18 Benchmark Functions
benchmarkFuncDic = {
    objectiveFunc.Ackley : "F10",
    objectiveFunc.Griewank : "F13",
    objectiveFunc.GeneralizedPenalizedFunc2 : "F12",
    objectiveFunc.Levy : "F14",
    objectiveFunc.Rastrigin : "F16",
    objectiveFunc.Schwefel : "F17",
    objectiveFunc.Schwefel220 : "F03",
    objectiveFunc.Schwefel221 : "F04",
    objectiveFunc.Schwefel222 : "F05",
    objectiveFunc.Schwefel223 : "F16",
    objectiveFunc.Schwefel226 : "F18",
    objectiveFunc.Quartic : "F15",
    objectiveFunc.RotatedHyperEllipsoid : "F02",
    objectiveFunc.Sphere : "F07",
    objectiveFunc.Step: "F08",
    objectiveFunc.SumSquares : "F09",
    objectiveFunc.Rosenbrock : "F01",
    objectiveFunc.GeneralizedPenalizedFunc1 : "F11"
}

AlgorithmsDic = {
    "GOA" : 0,
    "SPSO" : 4,
    "FiPSO" : 3,
    "BBPSO" : 2,
    "BBFiPSO" : 1
}

def benchmarkFunLabels(row):
    for funcName, funcF in benchmarkFuncDic.items():
        if funcName().functionName == row["ObjectiveFunction"]:
            return funcName().functionCode

def benchmarkFunType(row):
    for funcName, funcF in benchmarkFuncDic.items():
        if funcName().functionName == row["ObjectiveFunction"]:
            return funcName().fuctionMultiModal

def algorithmLabels(row):
    for algoName, algoIdx in AlgorithmsDic.items():
        if algoName == row["Algorithm"]:
            return algoIdx

# PATH to FILES
currentDir = "GOA_PSO_AllBenchmarkFuncs/2024-03-27_00-27_ANTS2024"
currentCSV = "Results_2024-03-27.csv"

plotsDir = "Plots"
NewDirPlots = os.path.join(currentDir, plotsDir)

try:
    os.mkdir(NewDirPlots)
except OSError as error:
    print(error)

# ============================================== Reading [CSV] Results =================================================
# Loading results
dfResultTemp = pd.read_csv(f"{currentDir}/{currentCSV}",
                       usecols= ["IDX", "BestObjective", "NumOfIndividuals", "NumOfDimensions", "NumOfIterations",
                                 "NumOfFunctionCall", "ElapsedTime", "ObjectiveFunction", "Algorithm"])
# ================================= ADDING FUNCTIONS' INFORMATION TO THE FRAMEWORK  ==================================
dfResult = dfResultTemp[dfResultTemp["Algorithm"].isin(AlgorithmsDic)].copy()
dfResult["ObjectiveFunctionCode"] = dfResult.apply(benchmarkFunLabels, axis=1)
dfResult["ObjectiveFunctionMultimodal"] = dfResult.apply(benchmarkFunType, axis=1)
dfResult["AlgorithmIdx"] = dfResult.apply(algorithmLabels, axis=1)

dfResult.sort_values(["ObjectiveFunctionCode", "AlgorithmIdx"], ascending=True, inplace=True)
# ======================================================================================================================
fontSize = 22
# ==================================== PLOTTING ALL TOGETHER (WITH OUTLIERS) ===========================================
# Set the overall figure size
plt.figure(figsize=(20, 40))

# Create a grid of plots with 3 columns and 6 rows
n_rows = 6
n_cols = 3
for i, benchmark_func in enumerate(dfResult["ObjectiveFunction"].unique()):
    benchmarkFuncCode = (dfResult[dfResult["ObjectiveFunction"]==benchmark_func].iloc[0])["ObjectiveFunctionCode"]
    plt.subplot(n_rows, n_cols, i+1)  # Creates subplot for each benchmark function
    sns.boxplot(x="Algorithm", y="BestObjective", data=dfResult[dfResult["ObjectiveFunction"] == benchmark_func])
    plt.title(f"{benchmarkFuncCode}: {benchmark_func}", fontweight='bold', fontsize=fontSize) # plt.title('Day vs. Total Bill', fontsize=16, fontweight='bold', fontname='Arial')
    plt.xticks(rotation=45, fontsize=fontSize, fontweight='bold')  # Rotates the algorithm names for better visibility

plt.tight_layout()

# Save the plot as an PNG file
plt.savefig(f"{NewDirPlots}/benchmark_comparison_plot.png", format='png', dpi=1200)
# Save the plot as an EPS file
plt.savefig(f"{NewDirPlots}/benchmark_comparison_plot.eps", format='eps', dpi=1200)
plt.clf()

# =================================== PLOTTING ALL TOGETHER (WITHOUT OUTLIERS) =========================================
# Set the overall figure size
plt.figure(figsize=(20, 40))

# Create a grid of plots with 3 columns and 6 rows
n_rows = 6
n_cols = 3
for i, benchmark_func in enumerate(dfResult["ObjectiveFunction"].unique()):
    benchmarkFuncCode = (dfResult[dfResult["ObjectiveFunction"]==benchmark_func].iloc[0])["ObjectiveFunctionCode"]
    plt.subplot(n_rows, n_cols, i+1)  # Creates subplot for each benchmark function
    sns.boxplot(x="Algorithm", y="BestObjective", data=dfResult[dfResult["ObjectiveFunction"] == benchmark_func], showfliers=False)
    plt.title(f"{benchmarkFuncCode}: {benchmark_func}", fontweight='bold', fontsize=fontSize) # plt.title('Day vs. Total Bill', fontsize=16, fontweight='bold', fontname='Arial')
    plt.xticks(rotation=45, fontsize=fontSize, fontweight='bold')  # Rotates the algorithm names for better visibility

plt.tight_layout()

# Save the plot as an PNG file
plt.savefig(f"{NewDirPlots}/benchmark_comparison_plot_NoOutlier.png", format='png', dpi=1200)
# Save the plot as an EPS file
plt.savefig(f"{NewDirPlots}/benchmark_comparison_plot_NoOutlier.eps", format='eps', dpi=1200)
plt.clf()

# ====================================== PLOTTING SEPERATELY (WITH OUTLIERS) ===========================================
# **************************************** (1) UNI-MODAL ***************************************************************
# Create a grid of plots with 3 columns and 3 rows (Total number of unimodal functions: 8)
# Set the overall figure size
plt.figure(figsize=(20, 20))

n_rows = 3
n_cols = 3
for i, benchmark_func in enumerate(dfResult.loc[dfResult["ObjectiveFunctionMultimodal"] == False, "ObjectiveFunction"].unique()):
    benchmarkFuncCode = (dfResult[dfResult["ObjectiveFunction"] == benchmark_func].iloc[0])["ObjectiveFunctionCode"]
    plt.subplot(n_rows, n_cols, i+1)  # Creates subplot for each benchmark function
    sns.boxplot(x="Algorithm", y="BestObjective", data=dfResult[dfResult["ObjectiveFunction"] == benchmark_func])
    plt.title(f"{benchmarkFuncCode}: {benchmark_func}", fontweight='bold', fontsize=fontSize)

    plt.xlabel("", fontweight='bold', fontsize=fontSize) # Algorithm
    plt.ylabel("Best Objective", fontweight='bold', fontsize=fontSize)
    if i % n_cols != 0:
        plt.ylabel('')  # Hide y-label for all but the first column subplots

    plt.xticks(rotation=45, fontsize=fontSize, fontweight='bold')  # Rotates the algorithm names for better visibility

plt.tight_layout()

# Save the plot as PNG and EPS files
plt.savefig(f"{NewDirPlots}/Unimodal_benchmark_comparison_plot.png", format='png', dpi=1200)
plt.savefig(f"{NewDirPlots}/Unimodal_benchmark_comparison_plot.eps", format='eps', dpi=1200)
plt.clf()

# **************************************** (2) MULTI-MODAL *************************************************************
# Create a grid of plots with 3 columns and 3 rows (Total number of multimodal functions: 9)
# Set the overall figure size
plt.figure(figsize=(20, 20))
n_rows = 3
n_cols = 3
for i, benchmark_func in enumerate(dfResult.loc[dfResult["ObjectiveFunctionMultimodal"] == True, "ObjectiveFunction"].unique()):
    benchmarkFuncCode = (dfResult[dfResult["ObjectiveFunction"] == benchmark_func].iloc[0])["ObjectiveFunctionCode"]
    plt.subplot(n_rows, n_cols, i+1)  # Creates subplot for each benchmark function
    sns.boxplot(x="Algorithm", y="BestObjective", data=dfResult[dfResult["ObjectiveFunction"] == benchmark_func])
    plt.title(f"{benchmarkFuncCode}: {benchmark_func}", fontweight='bold', fontsize=fontSize)

    plt.xlabel("", fontweight='bold', fontsize=fontSize) # Algorithm
    plt.ylabel("Best Objective", fontweight='bold', fontsize=fontSize)

    if i % n_cols != 0:
        plt.ylabel('')  # Hide y-label for all but the first column subplots

    plt.xticks(rotation=45, fontsize=fontSize, fontweight='bold')  # Rotates the algorithm names for better visibility

plt.tight_layout()

# Save the plot as PNG and EPS files
plt.savefig(f"{NewDirPlots}/Multimodal_benchmark_comparison_plot.png", format='png', dpi=1200)
plt.savefig(f"{NewDirPlots}/Multimodal_benchmark_comparison_plot.eps", format='eps', dpi=1200)
plt.clf()
# plt.show()

# ===================================== PLOTTING SEPERATELY (WITHOUT OUTLIERS) =========================================
# **************************************** (1) UNI-MODAL ***************************************************************
# Create a grid of plots with 3 columns and 3 rows (Total number of unimodal functions: 8)
# Set the overall figure size
plt.figure(figsize=(20, 20))

n_rows = 3
n_cols = 3
for i, benchmark_func in enumerate(dfResult.loc[dfResult["ObjectiveFunctionMultimodal"] == False, "ObjectiveFunction"].unique()):
    benchmarkFuncCode = (dfResult[dfResult["ObjectiveFunction"] == benchmark_func].iloc[0])["ObjectiveFunctionCode"]
    plt.subplot(n_rows, n_cols, i+1)  # Creates subplot for each benchmark function
    sns.boxplot(x="Algorithm", y="BestObjective", data=dfResult[dfResult["ObjectiveFunction"] == benchmark_func], showfliers=False)
    plt.title(f"{benchmarkFuncCode}: {benchmark_func}", fontweight='bold', fontsize=fontSize)

    plt.xlabel("", fontweight='bold', fontsize=fontSize) # Algorithm
    plt.ylabel("Best Objective", fontweight='bold', fontsize=fontSize)

    if i % n_cols != 0:
        plt.ylabel('')  # Hide y-label for all but the first column subplots

    plt.xticks(rotation=45, fontsize=fontSize, fontweight='bold')  # Rotates the algorithm names for better visibility

plt.tight_layout()

# Save the plot as PNG and EPS files
plt.savefig(f"{NewDirPlots}/Unimodal_benchmark_comparison_plot_NoOutlier.png", format='png', dpi=1200)
plt.savefig(f"{NewDirPlots}/Unimodal_benchmark_comparison_plot_NoOutlier.eps", format='eps', dpi=1200)
plt.clf()

# **************************************** (2) MULTI-MODAL *************************************************************
# Create a grid of plots with 3 columns and 3 rows (Total number of multimodal functions: 9)
# Set the overall figure size
plt.figure(figsize=(20, 20))
n_rows = 3
n_cols = 3
for i, benchmark_func in enumerate(dfResult.loc[dfResult["ObjectiveFunctionMultimodal"] == True, "ObjectiveFunction"].unique()):
    benchmarkFuncCode = (dfResult[dfResult["ObjectiveFunction"] == benchmark_func].iloc[0])["ObjectiveFunctionCode"]
    plt.subplot(n_rows, n_cols, i+1)  # Creates subplot for each benchmark function
    sns.boxplot(x="Algorithm", y="BestObjective", data=dfResult[dfResult["ObjectiveFunction"] == benchmark_func], showfliers=False)
    plt.title(f"{benchmarkFuncCode}: {benchmark_func}", fontweight='bold', fontsize=fontSize)

    plt.xlabel("", fontweight='bold', fontsize=fontSize) # Algorithm
    plt.ylabel("Best Objective", fontweight='bold', fontsize=fontSize)

    if i % n_cols != 0:
        plt.ylabel('')  # Hide y-label for all but the first column subplots

    plt.xticks(rotation=45, fontsize=fontSize, fontweight='bold')  # Rotates the algorithm names for better visibility

plt.tight_layout()

# Save the plot as PNG and EPS files
plt.savefig(f"{NewDirPlots}/Multimodal_benchmark_comparison_plot_NoOutlier.png", format='png', dpi=1200)
plt.savefig(f"{NewDirPlots}/Multimodal_benchmark_comparison_plot_NoOutlier.eps", format='eps', dpi=1200)
plt.clf()
