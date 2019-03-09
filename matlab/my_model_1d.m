close all;clear;clc;

% my model 2019.3.5

k1 = 1000;
k2 = 0;
m1 = 2;
m2 = 1;
m3 = 10;
c1 = 10;

%%
C = 0:0.1:30;
c_sz = size(C);
meet = zeros(c_sz);

for k = 1:c_sz(2)
    fprintf('%d/%d\n',k,c_sz(2));
    c2 = C(k);
    tspan=0:0.0001:5;
    y0=[0,0.2,1,0.1,0,0];
    [t,y]=ode45( @(t,y) odefun1(t,y,k1,k2,c1,c2,m1,m2,m3),tspan,y0);
    v_c = abs(y(:,4)-y(:,5));
    flag = 1;
    sz = size(tspan);
    interval = int64(sz(2)/50);
    for i = 1 : sz(2) - interval
        if sum(v_c(i:i+interval)) / double(interval) < 1e-4
            meet(k) = tspan(i);
            flag = 0;
            break;
        end
    end
    if flag == 1
        meet(k) = 4.9;
    end
end
plot(C,meet);

save data-5;

%%

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