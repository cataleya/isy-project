% conv layers Serie
home
close all


epochs(:,1) = pool_size3(:,1)
epochs(:,2) = epochs(:,1)
epochs(:,3) = epochs(:,1)


acc(:,1) = pool_size3(:,4)
acc(:,2) = pool_size5(:,4)
acc(:,3) = pool_size3nodrop(:,4)


figure(1)
plot(epochs(:,1), acc(:,1), 'LineWidth', 1.5)
hold on
plot(epochs(:,2), acc(:,2), 'LineWidth', 1.5)
hold on
plot(epochs(:,3), acc(:,3), 'LineWidth', 1.5)
hold off

title('Pool size', 'FontSize', 14)
xlabel('Epoche') 
ylabel('Erkennungsrate') 
legend('Pool size = 3', 'Pool size = 5', 'Pool size = 5, no dropout')
