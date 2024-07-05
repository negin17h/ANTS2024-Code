import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import researchpy as rp
from scipy import stats
import os

import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Visualization (Plotly)
import matplotlib.pyplot as plt
import seaborn as sns
import colorsys

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
    #"GOAwithN": 1,
    "SPSO" : 4,
    "FiPSO" : 3,
    "BBPSO" : 2,
    "BBFiPSO" : 1
}

# **********************************************************************
# Function to adjust the saturation of a given RGB color
def adjust_saturation(rgb, factor):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s = max(0, min(1, s * factor))  # Ensure saturation stays within [0, 1]
    return colorsys.hls_to_rgb(h, l, s)

paletteName = "deep"
base_palette = sns.color_palette(paletteName, 5)

# Increase the saturation by 50% (factor > 1)
# Reduce saturation by 50% (factor <1)
adjusted_palette = [adjust_saturation(color, 1.3) for color in base_palette]  

numOfColors = 7 # = Number of Algorithms
pal = sns.color_palette(paletteName, n_colors=numOfColors)
col = pal.as_hex()
# **********************************************************************

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
currentCSV = "Results_2024-03-27_00-27_SPSO.csv"

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
fontSize = 23
fontSizeOffset = 20
# ====================================== PLOTTING SEPERATELY (WITH OUTLIERS) ===========================================
# CM
width_cm = 50  # Desired width in centimeters
height_cm = 50  # Desired height in centimeters

width_in = width_cm / 2.54
height_in = height_cm / 2.54

plt.figure(figsize=(width_in, height_in))
# **************************************** (1) UNI-MODAL ***************************************************************

n_rows = 3
n_cols = 3
for i, benchmark_func in enumerate(dfResult.loc[dfResult["ObjectiveFunctionMultimodal"] == False, "ObjectiveFunction"].unique()):
    benchmarkFuncCode = (dfResult[dfResult["ObjectiveFunction"] == benchmark_func].iloc[0])["ObjectiveFunctionCode"]
    plt.subplot(n_rows, n_cols, i+1)  # Creates subplot for each benchmark function
    ax = sns.boxplot(x="Algorithm", y="BestObjective", palette=adjusted_palette, hue="Algorithm", data=dfResult[dfResult["ObjectiveFunction"] == benchmark_func])
    plt.title(f"{benchmarkFuncCode}: {benchmark_func}", fontweight='bold', fontsize=fontSize)

    plt.xlabel("", fontweight='bold', fontsize=fontSize) # Algorithm
    plt.ylabel("Best Objective", fontweight='bold', fontsize=fontSize)
    # # Increase the font size of the axis ticks
    # plt.tick_params(axis='both', which='major', labelsize=fontSize)
    # Increase the font size of the y-axis scale
    ax.yaxis.get_offset_text().set_fontsize(fontSizeOffset)

    # if i < (n_rows * n_cols) - n_cols:
    #     plt.xlabel('')  # Hide x-label for all but bottom row subplots
    if i % n_cols != 0:
        plt.ylabel('')  # Hide y-label for all but the first column subplots

    plt.xticks(rotation=45, fontsize=fontSize, fontweight='bold')  # Rotates the algorithm names for better visibility
    plt.yticks(fontsize=fontSize) #, fontweight='bold')

plt.tight_layout()

# Save the plot as PNG and EPS files
plt.savefig(f"{NewDirPlots}/Unimodal_benchmark_comparison_plot.png", format='png', dpi=1200)
plt.savefig(f"{NewDirPlots}/Unimodal_benchmark_comparison_plot.eps", format='eps', dpi=1200)
plt.clf()
# plt.show()

# **************************************** (2) MULTI-MODAL *************************************************************
# Create a grid of plots with 3 columns and 3 rows (Total number of multimodal functions: 9)
# Set the overall figure size
plt.figure(figsize=(width_in, height_in))
n_rows = 3
n_cols = 3
for i, benchmark_func in enumerate(dfResult.loc[dfResult["ObjectiveFunctionMultimodal"] == True, "ObjectiveFunction"].unique()):
    benchmarkFuncCode = (dfResult[dfResult["ObjectiveFunction"] == benchmark_func].iloc[0])["ObjectiveFunctionCode"]
    plt.subplot(n_rows, n_cols, i+1)  # Creates subplot for each benchmark function
    ax = sns.boxplot(x="Algorithm", y="BestObjective", palette=adjusted_palette, hue="Algorithm", data=dfResult[dfResult["ObjectiveFunction"] == benchmark_func])
    plt.title(f"{benchmarkFuncCode}: {benchmark_func}", fontweight='bold', fontsize=fontSize)

    plt.xlabel("", fontweight='bold', fontsize=fontSize) # Algorithm
    plt.ylabel("Best Objective", fontweight='bold', fontsize=fontSize)
    # # Increase the font size of the axis ticks
    # plt.tick_params(axis='both', which='major', labelsize=fontSize)
    # Increase the font size of the y-axis scale
    ax.yaxis.get_offset_text().set_fontsize(fontSizeOffset)

    # if i < (n_rows * n_cols) - n_cols:
    #     plt.xlabel('')  # Hide x-label for all but bottom row subplots
    if i % n_cols != 0:
        plt.ylabel('')  # Hide y-label for all but the first column subplots

    plt.xticks(rotation=45, fontsize=fontSize, fontweight='bold')  # Rotates the algorithm names for better visibility
    plt.yticks(fontsize=fontSize) #, fontweight='bold')

plt.tight_layout()

# Save the plot as PNG and EPS files
plt.savefig(f"{NewDirPlots}/Multimodal_benchmark_comparison_plot.png", format='png', dpi=1200)
plt.savefig(f"{NewDirPlots}/Multimodal_benchmark_comparison_plot.eps", format='eps', dpi=1200)
plt.clf()
# plt.show()
