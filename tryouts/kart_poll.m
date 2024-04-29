%script kart-poll

close all; clear
%graphics_toolkit ("gnuplot")
%set up

   %P'=f(x,p)
   %P=[x  o
   %   x' o']

   mk=1
   mp=1
   M=mk+mp;
   l=3

   f=@ (Pi) [-mp/M*l*cos(Pi(1,2))*Pi(2,2) , Pi(2,2) ; 0 , -sin(2*Pi(1,2)) *Pi(2,2)^2 /(2*M) - 9.8*sin(Pi(1,2))/l];  %g = 9.8
   ci=[0,pi/2;0,0]; %condiciones iniciales [x,y;x',y']
   %se empieza en t=0
   dt=0.01; %paso

%resolucion y grafico
P(:,:,1)=ci;

figure(1);hold on
axis([-1 3 -pi pi])
%runge kutta 4
for i=1:3
    k1=f(P(:,:,i));
    k2=f(P(:,:,i)+k1*dt/2);
    k3=f(P(:,:,i)+k2*dt/2);
    k4=f(P(:,:,i)+k3*dt);
    P(:,:,i+1)= P(:,:,i)+dt/6*(k1+2*k2+2*k3+k4);

    x(1,:)=P(1,1,[i:i+1]);
    y(1,:)=P(1,2,[i:i+1]);
    plot(x,y);
    pause(dt/2);
end%for

tol=1e-10;
i=4;
t=dt*4;
while t<5  % && i<5000
    %predictor: adam-bashborth
    P(:,:,i+1)=P(:,:,i)+dt/24 * (55*f(P(:,:,i))-59*f(P(:,:,i-1))+37*f(P(:,:,i-2))-9*f(P(:,:,i-3)));
    e=1;
    m=0;
    %corrector: adam-moulton
    while e>tol && m<50
        vm=P(:,:,i)+dt/24*(19*f(P(:,:,i))-5*f(P(:,:,i-1))+f(P(:,:,i-2))+9*f(P(:,:,i+1)));
        e=norm(vm-P(:,:,i+1));
        m=m+1;
        P(:,:,i+1)=vm;
    end%while

    x(1,:)=P(1,1,[i:i+1]);
    y(1,:)=P(1,2,[i:i+1]);
    plot(x,y);
    pause(dt/2);

    i=i+1;
    t=t+dt;
end%while
