% conv layers Serie
home
close all


epochs(:,1) = conv_layers_1(:,1)
epochs(:,2) = epochs(:,1)
epochs(:,3) = epochs(:,1)
epochs(:,4) = epochs(:,1)
epochs(:,5) = epochs(:,1)
epochs(:,6) = epochs(:,1)

acc(:,1) = conv_layers_1(:,4)
acc(:,2) = conv_layers_1nodrop(:,4)
acc(:,3) = conv_layers_3(:,4)
acc(:,4) = conv_layers_4(:,4)
acc(:,5) = conv_layers_5(:,4)
acc(:,6) = conv_layers_5nodrop(:,4)


figure(1)
plot(epochs(:,1), acc(:,1), 'LineWidth', 1.5)
hold on
plot(epochs(:,2), acc(:,2), 'LineWidth', 1.5)
hold on
plot(epochs(:,3), acc(:,3), 'LineWidth', 1.5)
hold on
plot(epochs(:,4), acc(:,4), 'LineWidth', 1.5)
hold on
plot(epochs(:,5), acc(:,5), 'LineWidth', 1.5)
hold on
plot(epochs(:,6), acc(:,6), 'LineWidth', 1.5)
hold off
title('Anzahl hintereinanderliegender Convolution Layers', 'FontSize', 14)
xlabel('Epoche') 
ylabel('Erkennungsrate') 
legend('1 Layer', '1 Layer, no dropout', '3 Layer', '4 Layer', '5 Layer', '5 Layer, no dropout')
