x1 = 200.0;
y1 = 5.0;
x2 = 2000.0;
y2 = 50.0;

m = (y2 - y1)/(x2 - x1);


x = 1800*rand(100,1) + 200;


%x = linspace(x1, x2, 100)';

yIdeal  = m * (x - x1) + y1;

uncertainty = 5;
yrand = uncertainty * rand(size(x))-uncertainty / 2;
yActual = yIdeal + yrand;
plot(x, yIdeal, x, yActual, '+');
%, x, yrand, '+')


t0 = 0;
t1 = .025;
h = t0 + t1 * x;
error = 0.5/length(x)*sum((h - y).^2)

save houseData
a=[
       1361.7,       31.857
         1815,       42.914
       1788.2,       46.076
       803.31,       20.993
        968.6,       22.318
       1646.8,       42.266
       1861.6,       44.849
       1760.1,       42.103
       1099.4,        28.39
       1252.1,       29.526
       1864.6,       44.403
       499.01,        10.65
       873.85,        19.42
       666.93,       15.119
       1694.9,       40.226
       326.32,       5.8472
       705.85,       16.686
       1199.1,       30.056
       1314.1,       32.321
       937.62,       25.916
       665.17,       16.327
       950.14,       22.011
        887.7,       23.583
        373.4,       10.541
       436.31,        8.846
       1297.8,       32.298
       1381.1,       33.493
       1713.8,       42.379
       1099.7,       27.143
       752.96,        20.68
       505.58,       13.676
       287.78,       4.7659
       606.47,       12.785
       860.55,       20.199
       1222.1,       31.044
       1225.5,       32.768
       1601.8,       39.217
       1262.4,       31.276
       399.19,       7.9755
       1353.2,       34.906
       329.63,       9.3491
       858.52,       22.875
       717.15,       20.139
       1534.6,       36.103
       320.58,       9.5162
       1562.9,       38.906
       633.28,       14.728
       615.64,       13.115
       221.58,       6.9772
       438.45,       12.957
        576.9,       15.801
       585.53,       16.325
       1830.4,       44.983
       699.99,       15.956
       374.78,       8.7618
       1609.1,       39.009
       1401.5,       32.646
       1302.8,       31.537
       818.83,       19.173
       1442.6,       34.578
       620.41,       14.751
       754.45,       19.047
       843.14,         19.2
       590.06,       13.215
       1419.1,       35.421
       1524.8,       37.656
       218.03,       6.6254
       840.65,       20.016
       631.92,       17.334
       1869.7,       48.811
       1573.4,        40.66
        882.9,       21.243
       1434.7,       36.965
         1458,       34.117
       1262.7,       32.146
       1931.7,       46.353
       1606.4,       37.914
       1169.2,       27.005
       369.45,       6.8673
       1794.8,       47.148
       663.49,       16.713
       1034.3,       27.139
       795.44,       20.335
       1562.2,       40.249
       1645.1,       40.225
       1645.6,       42.881
       1777.9,       44.417
       357.45,       7.0105
       1402.2,       36.617
       1104.4,       28.443
       804.01,       19.509
       1259.4,       32.579
       833.48,        20.43
       572.09,       12.082
       671.88,       17.222
       377.57,       8.4334
       470.11,       11.301
       777.83,       19.876
       1269.9,       30.826
       476.73,       10.155];