function forcG = whitenoiseWave(dt,t_max,Hs,Tw,s,wfreq,Fesys,DD)
%function [waveG,forcG,timevec,fnin,Hs2i] = whitenoiseWave(dt,t_max,Hs,Tw,s,wfreq,Fesys,DD)
%    [waveG,forcG,timevec,freq] = whitenoiseWave(dt,t_max,Hs,Tw,s,wfreq,Fesys)
Fs              = 1/dt;
Nl              = ceil(t_max*Fs);


rng(DD)
nin             = s*Hs*randn(Nl,1);
[~,fnin,NIN]    = espectro(nin,Fs);

[~,Hs2i,~,~]    = JONSWAP(2*pi*fnin,Tw,Hs,3.3);
WECFe           = interp1(wfreq,Fesys,2*pi*fnin,'linear'); %X,Y,Xq
    
WECFe(isnan(WECFe)) = 0;

[fid,msg] = fopen('WECFe.txt','wt');
assert(fid>=3,msg)
fprintf(fid,'%d %d\n',[real(WECFe);imag(WECFe)]);
fclose(fid);

Fspe                = NIN.'.*Hs2i.*WECFe;
Wspe                = NIN.'.*Hs2i;

[waveG,timevec] = fftInvertion(Wspe,dt,t_max);
[forcG,~]       = fftInvertion(Fspe,dt,t_max);
cte   = Hs/(4*std(waveG));
waveG = waveG.'*cte;
forcG = forcG.'*cte;