clear all, close all, clc

%simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
m=0.8; c=2.2; k=5.5; b=1.5;

%MSdamper = @(t,x)([ x(2)        ;...
%                    (1/m) * (-c*x(2)*abs(x(2)) - k*x(1) - b*(x(1)^3) + 0)]);


MSdamper = @(t,x,u)([ x(2)        ;...
                    (1/m) * (-c*x(2)*abs(x(2)) - k*x(1) - b*(x(1)^3) + u)]);



ode_options = odeset('RelTol',1e-10,'AbsTol',1e-11);

input=[]; output=[];
for j=1:100     %training trajectories
    x0=3*(rand(2,1)-0.5);
    u=2*(rand(1,1)-0.5);
    [t,y] = ode45(@(t,x) MSdamper(t,x,u),t,x0);
    input=[input; y(1:end-1,:) u*ones(length(t)-1,1)];
    output=[output; y(2:end,:)];
    plot3(y(:,1),y(:,2), u*ones(size(y(:,1))), 'Color', [0 (0.447-j/300) (0.741-j/300)]),hold on
    plot3(x0(1),x0(2), u,'ro','Color',[0 (0.447-j/300) (0.741-j/300)])
end

grid on

%% 

net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');

%%

for m = 1:1%0
    x0=3*(rand(2,1)-0.5);
    u = 2*(rand(1,1)-0.5);
    [t,y] = ode45(@(t,y) MSdamper(t,y,u),t,x0);
    
    ynn(:,1) = x0;
    for k = 2:(length(t))
        ynn(:,k) = net([ynn(:,k-1); u]);
    end
    
    
    for k = 1:length(ynn)
        figure(2)
        plot(y(1:k,1),y(1:k,2)); hold on
        plot(ynn(1,1:k),ynn(2,1:k));
        plot(x0(1),x0(2),'ro'); 
        plot(y(k,1),y(k,2),'.','Color','#0072BD','MarkerSize',15);
        plot(ynn(1,k),ynn(2,k),'.','Color','#D95319','MarkerSize',15);hold off
        pause(0.01)
    end
    
    
    figure(3)
    subplot(2,2,1), plot(t(1:end),y(1:end,1)), hold on, plot(t(1:end),ynn(1,:)')
    subplot(2,2,2), plot(t(1:end),y(1:end,2)), hold on, plot(t(1:end),ynn(2,:)')
    subplot(2,2,4), plot(t(1:end),y(1:end,1)-ynn(1,:)'+y(1:end,2)-ynn(2,:)')
end