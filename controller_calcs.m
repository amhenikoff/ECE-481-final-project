clear 
clc
format long

Ad = [1 .1 .005; 0 1 .1; 0 0 1];
Bd = [1/6000; .005; .1];
Cd = [1 0 0];
Dd = 0;

p1 = -0.6;
p2 = -0.1;
p3 = 0.8;

% desired_char_poly = conv(conv([1, -p1], [1, -p2]), [1, -p3]); % Gives [1, c2, c1, c0]
% c2 = desired_char_poly(2);
% c1 = desired_char_poly(3);
% c0 = desired_char_poly(4);

c0 = -p1*p2*p3;
c1 = p1*p2+p2*p3+p1*p3;
c2 = -(p1+p2+p3);

a0 = -1;
a1 = 3;
a2 = -3;

F_bar = [-c0+a0 -c1+a1 -c2+a2];
% V = [0.00016666667,0.0006666667,0.00016666667;
%      -0.005, 0, 0.005;
%      0.1, -0.2, 0.1];

C_AB = [Bd Ad*Bd Ad^2*Bd];
% disp(C_AB)

Ad_bar = [0 1 0;0 0 1; 1 -3 3];
Bd_bar = [0; 0; 1];
C_AB_bar = [Bd_bar Ad_bar*Bd_bar Ad_bar^2*Bd_bar];

V = C_AB*(C_AB_bar^-1);
% disp(V)

F = F_bar*(V^-1)

Gd = ss(Ad+(Bd*F),Bd,Cd-Dd*F,Dd,0.1);
step(Gd);
eig(Ad+Bd*F)

