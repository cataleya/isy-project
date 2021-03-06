% Pooling layer serie
home
close all


epochs(:,1) = neurons_in_dense_layer1024(:,1)
epochs(:,2) = epochs(:,1)
epochs(:,3) = epochs(:,1)
epochs(:,3) = epochs(:,1)
epochs(:,4) = epochs(:,1)
epochs(:,5) = epochs(:,1)
epochs(:,6) = epochs(:,1)


acc(:,1) = neurons_in_dense_layer8(:,4)
acc(:,2) = neurons_in_dense_layer16(:,4)
acc(:,3) = neurons_in_dense_layer64(:,4)
acc(:,4) = neurons_in_dense_layer256(:,4)
acc(:,5) = neurons_in_dense_layer1024(:,4)
acc(:,6) = neurons_in_dense_layer2048(:,4)


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

title('Variation der Neuronenzahl pro Layer', 'FontSize', 14)
xlabel('Epoche') 
ylabel('Erkennungsrate') 
legend('8 Neuronen', '16 Neuronen', '64 Neuronen', '256 Neuronen', '1024 Neuronen', '2048 Neuronen')
