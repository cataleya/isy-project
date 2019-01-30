% conv layers Serie
home
close all


figure(1)

% (gamma*<x, x'> + r)^d
scatter(poly2c0variiert(:,2), poly2c0variiert(:,3)) % gamma = 1
hold on
scatter(poly2c0variiert2(:,2), poly2c0variiert2(:,3)) % gamma = 0.004
hold off
legend('\gamma = 1.000', '\gamma = 0.004')
title('Variation von r', 'FontSize', 14)
xlabel('r') 
ylabel('Erkennungsrate') 


figure(2)
% (gamma*<x, x'> + r)^d
scatter(poly2gammakleinschrittigvariiert(:,1), poly2gammakleinschrittigvariiert(:,3)) % r = 0
hold on
scatter(poly2gammakleinschrittigvariiert2(:,1), poly2gammakleinschrittigvariiert2(:,3)) % r = 60
hold off
legend('r = 0', 'r = 60')
title('Variation von \gamma', 'FontSize', 14)
xlabel('\gamma', 'FontSize', 14) 
ylabel('Erkennungsrate') 
