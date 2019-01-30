% Pooling layer serie
home
close all


epochs(:,1) = pooling_layer1(:,1)
epochs(:,2) = epochs(:,1)
epochs(:,3) = epochs(:,1)


acc(:,1) = pooling_layer1(:,4)
acc(:,2) = pooling_layer1nodrop(:,4)
acc(:,3) = pooling_layer2(:,4)



figure(1)
plot(epochs(:,1), acc(:,1), 'LineWidth', 1.5)
hold on
plot(epochs(:,2), acc(:,2), 'LineWidth', 1.5)
hold on
plot(epochs(:,3), acc(:,3), 'LineWidth', 1.5)
hold off

title('Anzahl Pooling Layers', 'FontSize', 14)
xlabel('Epoche') 
ylabel('Erkennungsrate') 
legend('1 P-Layer', '1 P-Layer, no dropout', '2 P-Layer')
