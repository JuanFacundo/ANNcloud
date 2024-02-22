clear all, close all, clc

%simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=28;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ;...
                  r * x(1)-x(1)*x(3) - x(2) ;...
                  x(1) * x(2) - b*x(3)      ]);

ode_options = odeset('RelTol',1e-10,'AbsTol',1e-11);

input=[]; output=[];
for j=1:100     %training trajectories
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)),hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end

grid on, view(-23,18)

%% 
net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');

%%

for m = 1:1%0
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    
    ynn(:,1) = net(x0);
    for k = 2:(length(t)-1)
        ynn(:,k) = net(ynn(:,k-1));
    end

    figure(2)
    plot3(y(:,1),y(:,2),y(:,3)); hold on
    plot3(ynn(1,:),ynn(2,:),ynn(3,:));
    plot3(x0(1),x0(2),x0(3),'ro')
    
    figure(3), hold on
    plot(t(2:end),y(2:end,1)), plot(t(2:end),ynn(1,:)')
    %plot(t(2:end),y(2:end,1)-ynn(1,:)')
end