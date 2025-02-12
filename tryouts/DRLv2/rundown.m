clear all; clc;

load('F2V.mat')

dt      = 0.01;

A = F2V_model_p.A;
B = F2V_model_p.B;
C = F2V_model_p.C;
D = F2V_model_p.D;

ss = c2d(F2V_model_p,dt,'zoh');

Ad = ss.A;
Bd = ss.B;
Cd = ss.C;
Dd = ss.D;