% conv layers Serie
home
close all


figure(1)
% (gamma*<x, x'> + r)^d
semilogx(poly4heatmap(6:9,4), poly4heatmap(6:9,5), 'o') % gamma = 1
legend('\gamma = 1.0')
title('Variation von r', 'FontSize', 14)
xlabel('r') 
ylabel('Erkennungsrate') 


figure(2)
% (gamma*<x, x'> + r)^d
semilogx(poly4heatmap(1:5,3), poly4heatmap(1:5,5), 'o') % r = 1

legend('r = 1.0')
title('Variation von \gamma', 'FontSize', 14)
xlabel('\gamma', 'FontSize', 14) 
ylabel('Erkennungsrate') 
