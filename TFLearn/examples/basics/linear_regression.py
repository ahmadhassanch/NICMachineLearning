""" Linear Regression Example """

from __future__ import absolute_import, division, print_function

import tflearn

# Regression data
X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]
#X = [ 1361.7, 1815,  1788.2,  803.31,   968.6,  1646.8,  1861.6,  1760.1,  1099.4,  1252.1,  1864.6,  499.01,  873.85,  666.93,  1694.9,  326.32,  705.85,  1199.1,  1314.1,  937.62,  665.17,  950.14,   887.7,   373.4,  436.31,  1297.8,  1381.1,  1713.8,  1099.7,  752.96,  505.58,  287.78,  606.47,  860.55,  1222.1,  1225.5,  1601.8,  1262.4,  399.19,  1353.2,  329.63,  858.52,  717.15,  1534.6,  320.58,  1562.9,  633.28,  615.64,  221.58,  438.45,   576.9,  585.53,  1830.4,  699.99,  374.78,  1609.1,  1401.5,  1302.8,  818.83,  1442.6,  620.41,  754.45,  843.14,  590.06,  1419.1,  1524.8,  218.03,  840.65,  631.92,  1869.7,  1573.4,   882.9,  1434.7,    1458,  1262.7,  1931.7,  1606.4,  1169.2,  369.45,  1794.8,  663.49,  1034.3,  795.44,  1562.2,  1645.1,  1645.6,  1777.9,  357.45,  1402.2,  1104.4,  804.01,  1259.4,  833.48,  572.09,  671.88,  377.57,  470.11,  777.83,  1269.9,  476.73]
#Y = [31.857,42.914 ,46.076 ,20.993 ,22.318 ,42.266 ,44.849 ,42.103 , 28.39 ,29.526 ,44.403 , 10.65 , 19.42 ,15.119 ,40.226 ,5.8472 ,16.686 ,30.056 ,32.321 ,25.916 ,16.327 ,22.011 ,23.583 ,10.541 , 8.846 ,32.298 ,33.493 ,42.379 ,27.143 , 20.68 ,13.676 ,4.7659 ,12.785 ,20.199 ,31.044 ,32.768 ,39.217 ,31.276 ,7.9755 ,34.906 ,9.3491 ,22.875 ,20.139 ,36.103 ,9.5162 ,38.906 ,14.728 ,13.115 ,6.9772 ,12.957 ,15.801 ,16.325 ,44.983 ,15.956 ,8.7618 ,39.009 ,32.646 ,31.537 ,19.173 ,34.578 ,14.751 ,19.047 ,  19.2 ,13.215 ,35.421 ,37.656 ,6.6254 ,20.016 ,17.334 ,48.811 , 40.66 ,21.243 ,36.965 ,34.117 ,32.146 ,46.353 ,37.914 ,27.005 ,6.8673 ,47.148 ,16.713 ,27.139 ,20.335 ,40.249 ,40.225 ,42.881 ,44.417 ,7.0105 ,36.617 ,28.443 ,19.509 ,32.579 , 20.43 ,12.082 ,17.222 ,8.4334 ,11.301 ,19.876 ,30.826 ,10.155]

# Linear Regression graph
input_ = tflearn.input_data(shape=[None])
linear = tflearn.single_unit(input_)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=0.0001)
m = tflearn.DNN(regression)
m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)

print("\nRegression result:")
print("Y = " + str(m.get_weights(linear.W)) +
      "*X + " + str(m.get_weights(linear.b)))

print("\nTest prediction for x = 3.2, 3.3, 3.4:")
print(m.predict([3.2, 3.3, 3.4]))
# should output (close, not exact) y = [1.5315033197402954, 1.5585315227508545, 1.5855598449707031]