%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wavestar WEC simulation
% Last update: 15/11/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clearvars

% Wave-to-force model -----------------------------------------------------
% Input: Free-surface elevation (in m)
% Output: Wave excitation torque (in N/m)
W2F         = load('W2F','SYS_ok','tau_f');
W2F_model   = W2F.SYS_ok;
W2F_advance = W2F.tau_f;
w           = 0:0.01:50; 
Fe          = (squeeze(freqresp(W2F_model,w)).*exp(1i*w*W2F_advance).').';

% Force-to-velocity model -------------------------------------------------
% Input: Wave excitation torque (in N/m)
% Output: Device velocity (in rad/s)
F2V         = load('F2V','F2V_model_p');
F2V_model   = F2V.F2V_model_p;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt      = 0.01;                       % Simulation step
t_end   = 2000;                       % Time definition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wave generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input wave definition ---------------------------------------------------
Hw                  =  0.15/2;  % Significant wave height
Tw                  =  1.3;     % Peak wave period  
gamma               =  3.3;     % Peak-enhancement factor (Fix to 3.3!)
seed_realisation    =  1;       % Change this number to change seed generation (and obtain a new wave realisation)
[eta,Fex,tsim,~,~]  = whitenoiseWave(dt,t_end,Hw,Tw,1,w,Fe,seed_realisation);
% eta: Free-surface elevation (la ola)
% Fex: Wave excitation force (la fuerza)

% -------------------------------------------------------------------------

%[fid,msg] = fopen('fex.txt','wt');
%assert(fid>=3,msg)
%fprintf(fid,'%d\n',Fe);
%fclose(fid);

% -------------------------------------------------------------------------

% Simulation variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
input       = timeseries(Fex,tsim);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For comparison purposes: PI controller via impedance-matching
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Optimal unconstrained control -------------------------------------------
wI  = 2*pi/Tw;
G   = frd(F2V_model,w);
K   = 1/ctranspose(G);
T   = G*ctranspose(G)/(G + ctranspose(G));

% Reactive controller -----------------------------------------------------
K_wI    = freqresp(K,wI); 
k_P     = real(K_wI);                   % Interpolate real part with k_P
k_I     = -wI*imag(K_wI);               % Interpolate imag part with k_I
K_PI    = tf(k_P) + tf(k_I, [1 0]);     % Synthesis PI controller
T_PI    = G/(1+G*K_PI);                 % PI controller input-output response

%%%%%%%%%%%%%%%%%%%%% RUN "wavestarSimulator.slx %%%%%%%%%%%%%%%%%%%%%%%%%%


