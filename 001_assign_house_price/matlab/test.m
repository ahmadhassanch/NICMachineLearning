x1 = 200;
y1 = 5;
x2 = 2000;
y2 = 50;

m = (y2 - y1)/(x2 - x1);


x = 1800*rand(100,1) + 200;


%x = linspace(x1, x2, 100)';

yIdeal  = m * (x - x1) + y1;

uncertainty = 5;
yrand = uncertainty * rand(size(x))-uncertainty / 2;
yActual = yIdeal + yrand;
plot(x, yIdeal, x, yActual, '+');
%, x, yrand, '+')

save houseData
