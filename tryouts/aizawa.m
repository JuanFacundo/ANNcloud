clear all, close all, clc

%simulate Aizawa system
dt=0.01; T=50; t=0:dt:T;
alf = 0.95; bet = 0.7; del = 3.5; eps = 0.25; gam = 0.6; eta = 0.1;

Aizawa = @(t,x)([ (x(3)-bet)*x(1)-del*x(2)  ;...
                  del*x(1)+(x(3)-bet)*x(2)     ;...
                  gam + alf*x(3) - (1/3)*(x(3)^3) - (x(1)^2) + eta*x(3)*(x(1)^3)    ]);

ode_options = odeset('RelTol',1e-10,'AbsTol',1e-11);

input=[]; output=[];
for j=1:100     %training trajectories
    x0=(1e-2)*(rand(3,1)-0.5);
    [t,y] = ode45(Aizawa,t,x0);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3), 'Color', [0 (0.447-j/300) (0.741-j/300)]),hold on
    plot3(x0(1),x0(2),x0(3),'ro','Color',[0 (0.447-j/300) (0.741-j/300)])
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
    x0=[-0.004129892948569;-0.001569909852891;0.003517219784020]
    [t,y] = ode45(Aizawa,t,x0);
    
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