clear all, close all, clc

%simulate CART and POLE system
dt=0.01; T=10; t=0:dt:T;
m1 = 10; m2 = 5; L = 0.4; g = 9.8; M = m1 + m2;

cartPole = @(t,x)([ m2*L*cos(x(2))*x(3)/M                                 ;...
                    x(3)                                                ;...
                    -((((x(3)^2)/(2*M))*sin(2*x(2)))+(g/L)*sin(x(2)))       ]);

ode_options = odeset('RelTol',1e-10,'AbsTol',1e-11);

input=[]; output=[];
for j=1:100     %training trajectories
    x0=2*(rand(3,1)-0.5);
    [t,y] = ode45(cartPole,t,x0);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3), 'Color', [0 (0.447-j/300) (0.741-j/300)]),hold on
    plot3(x0(1),x0(2),x0(3),'ro','Color',[0 (0.447-j/300) (0.741-j/300)])
end

grid on

%% 
net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'poslin';
net = train(net,input.',output.');

%%

for m = 1:1%0
    %x0=[-0.004129892948569;-0.001569909852891;0.003517219784020]
    x0=2*(rand(3,1)-0.5);
    [t,y] = ode45(cartPole,t,x0);
    
    ynn(:,1) = x0;
    for k = 2:(length(t))
        ynn(:,k) = net(ynn(:,k-1));
    end
    
%        figure(2)
%        plot3(y(:,1),y(:,2),y(:,3)); hold on
%        plot3(ynn(1,:),ynn(2,:),ynn(3,:));
%        plot3(x0(1),x0(2),x0(3),'ro')
    
    for k = 1:length(ynn)
        figure(2)
        plot3(y(1:k,1),y(1:k,2),y(1:k,3)); hold on
        plot3(ynn(1,1:k),ynn(2,1:k),ynn(3,1:k));
        plot3(x0(1),x0(2),x0(3),'ro'); 
        plot3(y(k,1),y(k,2),y(k,3),'.','Color','#0072BD','MarkerSize',15);
        plot3(ynn(1,k),ynn(2,k),ynn(3,k),'.','Color','#D95319','MarkerSize',15);hold off
        pause(0.01)
    end
    
    
    figure(3)
    subplot(2,2,1), plot(t(1:end),y(1:end,1)), hold on, plot(t(1:end),ynn(1,:)')
    subplot(2,2,2), plot(t(1:end),y(1:end,2)), hold on, plot(t(1:end),ynn(2,:)')
    subplot(2,2,3), plot(t(1:end),y(1:end,3)), hold on, plot(t(1:end),ynn(3,:)')
    subplot(2,2,4), plot(t(1:end),sqrt((y(1:end,1)-ynn(1,:)').^2+(y(1:end,2)-ynn(2,:)').^2+(y(1:end,3)-ynn(3,:)').^2))
end
