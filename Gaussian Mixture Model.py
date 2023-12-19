# -*- coding: utf-8 -*-
"""
Advanced Marketing Models Assignment

Created on Tue Nov 22 10:46:42 2022

@author: Carlos de Cloet
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def uniNormPDF(y: int, mu: int, sigma: int):
    pdf = 1/(2*math.pi)*math.exp(-1/2*(y-mu)**2/sigma)
    return pdf

def biNormPDF(y0: int, y1: int, mu0: int, mu1: int, stdev0: int, stdev1: int, rho: int):
    z = ((y0 - mu0)/stdev0)**2 - (2*rho*(y0-mu0)*(y1-mu1)/(stdev0*stdev1)) + ((y1-mu1)/stdev1)**2
    pdf = 1/(2*math.pi*stdev0*stdev1*math.sqrt(1-rho**2))*math.exp(-1/(2*(1-rho**2))*z)
    return pdf

def logBiNormPDF(y0: int, y1: int, mu0: int, mu1: int, stdev0: int, stdev1: int, rho: int):
    z = -1/(2*(1-rho**2))*(((y0 - mu0)/stdev0)**2 - (2*rho*(y0-mu0)*(y1-mu1)/(stdev0*stdev1)) + ((y1-mu1)/stdev1)**2)
    logpdf = -1*math.log(2*math.pi)-math.log(stdev0)-math.log(stdev1)-1/2*math.log(1-rho**2)+z
    return logpdf

def logL(mu: list, sigma: list, pi: list, y: list):
    
    loglikelihood = 0
    for i in range(len(y)):
        
        likelihood = 0
        for j in range(len(pi)):
        
            y0 = y[i][0]
            y1 = y[i][1]
            mu0 = mu[j][0]
            mu1 = mu[j][1]
            stdev0 = math.sqrt(sigma[j][0])
            stdev1 = math.sqrt(sigma[j][2])
            rho = sigma[j][1]/(stdev0*stdev1)
            segment = pi[j]
        
            likelihood += biNormPDF(y0, y1, mu0, mu1, stdev0, stdev1, rho)*segment           
        loglikelihood += math.log(likelihood)
        
    return loglikelihood

def EStep(mu: list, sigma: list, pi: list, y: list):
    
    conditionalProbMatrix = []
    
    for i in range(len(y)):        
        conditionalProbMatrix.append([])
        for j in range(len(pi)):
            y0 = y[i][0]
            y1 = y[i][1]
            mu0 = mu[j][0]
            mu1 = mu[j][1]
            stdev0 = math.sqrt(sigma[j][0])
            stdev1 = math.sqrt(sigma[j][2])
            rho = sigma[j][1]/(stdev0*stdev1)
            segment = pi[j]
            conditionalProbMatrix[i].append(logBiNormPDF(y0, y1, mu0, mu1, stdev0, stdev1, rho)*segment)
            
            
    for i in range(len(y)):
        LogSumExpProbabilities = math.log(sum(np.exp(conditionalProbMatrix[i])))
        for j in range(len(pi)):
            entry = conditionalProbMatrix[i][j]
            conditionalProb = entry - LogSumExpProbabilities
            conditionalProbMatrix[i][j] = math.exp(conditionalProb)
        
    return conditionalProbMatrix

def MStep(y: list, matrix: list):
    
    N = len(matrix)
    K = len(matrix[0]) ## prone to buggs
    segments = []  
    mu = []
    sigma = []
    
    for j in range(K): 
        sumForSegments = 0
        sumForMu0 = 0
        sumForMu1 = 0    
        
        for i in range(N):
            sumForSegments += matrix[i][j]   
            sumForMu0 += matrix[i][j]*y[i][0]
            sumForMu1 += matrix[i][j]*y[i][1]                     
        segments.append(sumForSegments/N)
        mu.append([sumForMu0/sumForSegments, sumForMu1/sumForSegments])
        
  
        
    
    for j in range(K):
        sumForSigma0 = 0
        sumForSigma1 = 0
        sumForSigma2 = 0
        for i in range(N):
            sumForSegments += matrix[i][j]
            sumForSigma0 += matrix[i][j]*(y[i][0] - mu[j][0])**2
            sumForSigma1 += matrix[i][j]*(y[i][0] - mu[j][0])*(y[i][1] - mu[j][1])
            sumForSigma2 += matrix[i][j]*(y[i][1] - mu[j][1])**2
        sigma.append([sumForSigma0/sumForSegments, sumForSigma1/sumForSegments, sumForSigma2/sumForSegments])
        
    return [mu, sigma, segments]

def generateStartMatrix(y: list, K: int):

    startMatrix = []
    
    rnd_nmr = np.random.sample(K)
    sum_rnd_nmr = np.sum(rnd_nmr)
    rnd_weight = rnd_nmr/sum_rnd_nmr
    
    for i in range(len(y)):
        startMatrix.append([])
        rnd_nmr = np.random.sample(K)
        sum_rnd_nmr = np.sum(rnd_nmr)
        rnd_weight = rnd_nmr/sum_rnd_nmr
        for j in range(K):
            startMatrix[i].append(rnd_weight[j])
    return startMatrix
        

def notConverged(newSegments: list, oldSegments: list):
    
    epsilon = 0.005
    booleanList = []
    
    for i in range(len(newSegments)):
        booleanList.append(abs(newSegments[i] - oldSegments[i]) < epsilon)
    return not(all(booleanList))


def EM(y: list, K: int):
    
    startMatrix = generateStartMatrix(y, K)
    
    mStepOutput = MStep(y, startMatrix) 
    mu = mStepOutput[0];
    sigma = mStepOutput[1]
    pi = mStepOutput[2]
    
    oldSegments = pi
    
    eStepOutput = EStep(mu, sigma, pi, y)
    mStepOutput=MStep(y,eStepOutput)
    
    mu = mStepOutput[0];
    sigma = mStepOutput[1]
    pi = mStepOutput[2]
    
    
    newSegments = mStepOutput[2]
    
    iteration = 2

    while(notConverged(newSegments, oldSegments)):
        oldSegments = newSegments
        
        eStepOutput = EStep(mu, sigma, pi, y)
        mStepOutput = MStep(y,eStepOutput)
        
        mu = mStepOutput[0];
        sigma = mStepOutput[1]
        pi = mStepOutput[2]
        newSegments = pi
        iteration +=  iteration + 1
        print('The iteration is: ' + str(iteration))
        
        if iteration > 50000:
            break
        
        
    return [mu, sigma, pi, iteration]

def ConditionalExpectation(y: int, mu: list, sigma: list, segments: list):
    conditionalExp = 0
    for i in range(len(segments)):
    
        rho = sigma[i][1]/(math.sqrt(sigma[i][0])*math.sqrt(sigma[i][2]))
        condExp = mu[i][0] + rho/sigma[i][0]*(y-mu[i][0])
        segmentValue = segments[i]*uniNormPDF(y, mu[i][0], sigma[i][0])
        summedSegmentValue = 0
        for j in range(len(segments)):
            summedSegmentValue += segments[j]*uniNormPDF(y, mu[j][0], sigma[j][0])
        conditionalExp += condExp*segmentValue/summedSegmentValue   
    return conditionalExp       

################################################################################################

################################################################################################

logLikelihoods = []


for i in range(10):
    print("Current iteration of random start is: " + str(i+1))
    
    random.seed(i)
    
    df = pd.read_csv(r'C:\Users\Carlos de Cloet\Desktop\Advanced Marketing Models\621810.csv')
    
    y = df.values.tolist()

    results = EM(y,3)

    mu = results[0]
    sigma = results[1]
    segments = results[2]
    
    logLikelihood = logL(mu,sigma,segments,y)
    logLikelihoods.append(logLikelihood)
    #print(logLikelihood)

maxValue = -9999999
index = 0
for i in range(len(logLikelihoods)):
    if logLikelihoods[i] > maxValue:
        maxValue = logLikelihoods[i]
        index = i

random.seed(index)

results = EM(y,3)

mu = results[0]
sigma = results[1]
segments = results[2]

logLikelihood = logL(mu,sigma,segments,y)
print(logLikelihood)
BIC = ((2 + 3 + 1)*len(segments)-1)*math.log(len(y)) - 2*logLikelihood
print(BIC)

y0 = []
y1 = []

for i in range(len(y)):
    y0.append(y[i][0])
    y1.append(y[i][1])
       

y0.sort()
 
y_min = y0[0]
y_max = y0[1999]

y_range = np.linspace(y_min, y_max, 1000)
predictions = []
for i in range(len(y_range)):
    predictions.append(ConditionalExpectation(y_range[i], mu, sigma, segments))
    
plt.scatter(y0,y1, s = 4)   
plt.plot(y_range,predictions, 'r', label ="Conditional expectation")
plt.legend()
plt.xlabel('y1')
plt.ylabel('y2')
plt.show()