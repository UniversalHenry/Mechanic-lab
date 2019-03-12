

clear;clc;
k1 = 1e11;
k2 = 0;
m1 = 800;
m2 = 10;
m3 = 200;
c1 = 10;
c2 = 1000;

%%
tspan=0:1e-6:10;
y0=[0,0.2,10,10,0,0];
[t,y]=ode45( @(t,y) odefun1(t,y,k1,k2,c1,c2,m1,m2,m3),tspan,y0);
subplot(121);
plot(t,y(:,4),t,y(:,5),t,y(:,6));
legend('v1','v2','v3');
subplot(122);
plot(t,y(:,2)-y(:,1));
legend('x2-x1');


function dx=odefun1(t,y,k1,k2,c1,c2,m1,m2,m3)
dx = zeros(6,1);   
dx(1:3) = y(4:6);
dis =  y(2) - y(1);
if dis < 0.2
    contactForce = k1*(dis - 0.2)-c1*(y(4)-y(5));
else
    contactForce = 0;
end
dx(4) = contactForce/m1;
dx(5) = (-contactForce-c2*(y(5)-y(6)))/m2;
dx(6) = (c2*(y(5)-y(6)))/m3;
end