import pandas as pd
import numpy as np

def writeCSV(features,yVector,csvName):
	data = pd.DataFrame(features)
	data.insert(features.shape[1],features.shape[1],yVector)
	data.to_csv(csvName, index=False, header=False)

def readCSV(path):
	data = pd.read_csv(path,header=None)
	features = np.array(data.iloc[:,:-1])
	target = np.array(data.iloc[:,-1])
	return features,target


