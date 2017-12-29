import numpy as np
import matplotlib.pyplot as plt
import linearRegressionModel as linRegModel

def generateDataPoly(xmin, xmax, m):
	x = np.linspace(xmin, xmax, m);
	np.random.shuffle(x)
	return x

def computeDataPoly(theta, x, noise):
	yIdeal = np.polyval(theta, x);
	random = np.random.rand(m);
	yrand = 2 * noise * (random - 0.5);     #random = [0, 1]
	y = yIdeal + yrand;
	return y, yIdeal;

def generateX(xRange, m):
	X = [];
	N = xRange.shape[0];
	for i in range(N):
		x = generateDataPoly(xRange[i,0], xRange[i,1], m);
		X.append(x);
	X = np.array(X);
	X = X.T;
	return X;

def computeY(theta, X, noise):
	#y, yIdeal = computeY(theta,X,noise)
	m = X.shape[0];
	N = X.shape[1];

	yIdeal = linRegModel.hypothesis(theta, X);
	random = np.random.rand(m);
	yRand = 2 * noise * (random - 0.5);     #random = [0, 1]
	y = yIdeal + yRand; 

	return y, yIdeal

def generateData(theta, xRange, m, noise):
	X  = generateX(xRange, m); #in excel sytle data (each row is a sample/example, contains multiple feature)
	y, yIdeal = computeY(theta, X, noise)
	return X, y, yIdeal




#plotIdealvsActual(x, y, yIdeal)
#x = [ 1361.7, 1815,  1788.2,  803.31,   968.6,  1646.8,  1861.6,  1760.1,  1099.4,  1252.1,  1864.6,  499.01,  873.85,  666.93,  1694.9,  326.32,  705.85,  1199.1,  1314.1,  937.62,  665.17,  950.14,   887.7,   373.4,  436.31,  1297.8,  1381.1,  1713.8,  1099.7,  752.96,  505.58,  287.78,  606.47,  860.55,  1222.1,  1225.5,  1601.8,  1262.4,  399.19,  1353.2,  329.63,  858.52,  717.15,  1534.6,  320.58,  1562.9,  633.28,  615.64,  221.58,  438.45,   576.9,  585.53,  1830.4,  699.99,  374.78,  1609.1,  1401.5,  1302.8,  818.83,  1442.6,  620.41,  754.45,  843.14,  590.06,  1419.1,  1524.8,  218.03,  840.65,  631.92,  1869.7,  1573.4,   882.9,  1434.7,    1458,  1262.7,  1931.7,  1606.4,  1169.2,  369.45,  1794.8,  663.49,  1034.3,  795.44,  1562.2,  1645.1,  1645.6,  1777.9,  357.45,  1402.2,  1104.4,  804.01,  1259.4,  833.48,  572.09,  671.88,  377.57,  470.11,  777.83,  1269.9,  476.73]
#y = [31.857,42.914 ,46.076 ,20.993 ,22.318 ,42.266 ,44.849 ,42.103 , 28.39 ,29.526 ,44.403 , 10.65 , 19.42 ,15.119 ,40.226 ,5.8472 ,16.686 ,30.056 ,32.321 ,25.916 ,16.327 ,22.011 ,23.583 ,10.541 , 8.846 ,32.298 ,33.493 ,42.379 ,27.143 , 20.68 ,13.676 ,4.7659 ,12.785 ,20.199 ,31.044 ,32.768 ,39.217 ,31.276 ,7.9755 ,34.906 ,9.3491 ,22.875 ,20.139 ,36.103 ,9.5162 ,38.906 ,14.728 ,13.115 ,6.9772 ,12.957 ,15.801 ,16.325 ,44.983 ,15.956 ,8.7618 ,39.009 ,32.646 ,31.537 ,19.173 ,34.578 ,14.751 ,19.047 ,  19.2 ,13.215 ,35.421 ,37.656 ,6.6254 ,20.016 ,17.334 ,48.811 , 40.66 ,21.243 ,36.965 ,34.117 ,32.146 ,46.353 ,37.914 ,27.005 ,6.8673 ,47.148 ,16.713 ,27.139 ,20.335 ,40.249 ,40.225 ,42.881 ,44.417 ,7.0105 ,36.617 ,28.443 ,19.509 ,32.579 , 20.43 ,12.082 ,17.222 ,8.4334 ,11.301 ,19.876 ,30.826 ,10.155]
#yp = hypothesis(t0, t1, x);
#plt.plot(x,y,'k+',x,yp)
#plt.grid()
#plt.show()
#error = 0.5/length(x)*sum((h - y)**2)
