function [waveG,forcG,timevec,fnin,Hs2i] = wavegen(dt,t_max,Hs,Tw,DD)
    %    [waveG,forcG,timevec,freq] = whitenoiseWave(dt,t_max,Hs,Tw,s,wfreq,Fesys)
    W2F         = load('W2F','SYS_ok','tau_f');
    W2F_model   = W2F.SYS_ok;
    W2F_advance = W2F.tau_f;
    wfreq       = 0:0.01:50; 
    Fesys       = (squeeze(freqresp(W2F_model,w)).*exp(1i*w*W2F_advance).').';
    
    Fs              = 1/dt;
    Nl              = ceil(t_max*Fs);
    
    
    rng(DD)
    nin             = s*Hs*randn(Nl,1);
    [~,fnin,NIN]    = espectro(nin,Fs);
    
    [~,Hs2i,~,~]    = JONSWAP(2*pi*fnin,Tw,Hs,3.3);
    WECFe           = interp1(wfreq,Fesys,2*pi*fnin,'linear'); %X,Y,Xq
        
    WECFe(isnan(WECFe)) = 0;
    
    Fspe                = NIN.'.*Hs2i.*WECFe;
    Wspe                = NIN.'.*Hs2i;
    
    [waveG,timevec] = fftInvertion(Wspe,dt,t_max);
    [forcG,~]       = fftInvertion(Fspe,dt,t_max);
    cte   = Hs/(4*std(waveG));
    waveG = waveG.'*cte;
    forcG = forcG.'*cte;
end

function [t,f,Y]=espectro(y,Fs)
    % [t,f,Y]=espectro(y,Fs)
    % y: signal
    % Fs: Sample rate
    % t: signal time
    % Y: Signal Fourier Spectrum
    T = 1/Fs;
    L = length(y);
    t = (0:L-1)*T;
    NFFT = 2^nextpow2(L);
    Y = fft(y,NFFT)/L;
    Y = Y(1:NFFT/2+1);
    f = Fs/2*linspace(0,1,NFFT/2+1);
end

function [rSt,timeinv] = fftInvertion(Fspe,dt,t_max)
    % [rSt,timeinv] = fftInvertion(Fspe,dt,t_max)
    
    Nl     = t_max/dt;
    f_fSt  = Fspe.';
    % f_fSt  = f_fSt.*(fnin.' <= 10 ) + 0.*(fnin.' > 10);
    if_fSt = [f_fSt(1:end-1);f_fSt(end-1:-1:1)];
    rSt    = real(ifft(if_fSt,'symmetric'));
    
    timeinv  = (0:length(rSt)-1)*dt; 
    [~,ddd] = min(abs(timeinv-t_max));
    rSt    = (rSt(1:ddd-1))*Nl;
    timeinv = timeinv(1:ddd-1);
end

function [S,Hw,Pwave,Te] = JONSWAP(w,Tp,Hs,gamma)
    nw=length(w);
    f=w/(2.*pi);
    A=0.3125*Hs^2/Tp^4;
    B=1.25/Tp^4;
    fp=1./Tp;
    m0=0.;
    m_1=0.;
    Pwave=0.;
    Hw=zeros(1,nw);
    S=zeros(1,nw);
    for i=2:nw
        fc=0.5*(f(i)+f(i-1));
        df=f(i)-f(i-1);
        S(i)=A/fc^5*exp(-B/fc^4);
        if (fc<fp)
            sigma=0.02;
            
        else
           sigma=0.25;
           
        end
        pa=exp(-(fc-fp)^2/(2.*sigma^2*fp^2));
        S(i)=S(i)*gamma^pa;
        m0=m0+S(i)*df;
        m_1=m_1+S(i)/fc*df;
    end
    alpha=Hs^2/(16.*m0);
    Te=m_1/m0;
    for i=2:nw
        df=f(i)-f(i-1);
        S(i)=S(i)*alpha;
        Hw(i)=sqrt(2.*S(i)*df);
        Pwave=Pwave+0.25*1025.*9.81*9.81*Hw(i)*Hw(i)/w(i);
    end
end



    
    