import numpy as np
import matplotlib.pyplot as plt
import linearRegressionModel as linRegModel
import CSVutils as csv



features,y = csv.readCSV('03_multiFeatureLinearReg.csv')

print(features)
print(y)
