close all;clear;clc;
tic; %comment%
experiment = 'b';
switch experiment
    case 'a' 
        a = 1:4:80; % scaling parameters
        b = 40*ones(1,length(a));
    case 'b'
        b = logspace(2,-3,20);        
        a = 1*ones(1,length(b));        
end

%% Gaussian mixture model generation
M = 40; % number of Gaussian mixtures
p = 10; %dimension of observation x
q = 10; % Dimension of data t
mu_m = randn(q,M); % Generating random mean vectors
mu_m = normc(mu_m); %Normalize columns of mu_m to have unit norm
T = sum(mu_m,2);
SNR = (1./b).*(q+a.^2-(a./M).^2.*trace(T*T')); % SNR based on different scaling parameters a
SNR_dB = 10.*log10(SNR);
Cm = zeros(q,q,M);
for i=1:M
    %C = sqrt(0.01)*randn(p);
    %Css(:,:,i) = C'*C; %covariance matrix for each Gaussian
    Cm(:,:,i) = eye(q); %
end
alpha = (1/M)*ones(M,1); %mixing proportions
Monte_Carlo = 300; % No.of simulations for evaluating optimal MSE
H = randn(p,q);
H = normc(H);
%% MSE evaluation of SSFN
ssfn_MSE = zeros(length(SNR),1);
optimal_MSE = zeros(length(SNR),1);
for k = 1:length(SNR)
    mu = a(k)*mu_m; % mean with scaling parameter a(k)
    gm = gmdistribution(mu',Cm,alpha); % Gaussian mixture model
    %% Data generation for SSFN
    sample = 3e3; % number of data points
    t = random(gm,sample); % draw random signals from GMM
    t = t';
    H = randn(p,q);
    H = normc(H);
    x = zeros(p,sample);
    for i=1:sample
        n = sqrt(b(k)/p)*randn(p,1); %Zero mean Gaussian noise samples
        x(:,i) = H*t(:,i) + n; % noisy signal generation
    end
    x = x'; t = t';
    idx = (randperm(sample)<=sample*0.7);
    [~, ssfn_MSE(k)] = ssfn_estimator(x(idx,:)',t(idx,:)',x(~idx,:)',t(~idx,:)');
    
    %% MSE evaluation of MMSE estimator
            
        t = random(gm,Monte_Carlo);
        t = t';
    parfor iter = 1:Monte_Carlo
        H = randn(p,q);
        H = normc(H);        
        n = sqrt(b(k)/p)*randn(p,1); %Zero mean Gaussian noise samples
        
        x = H*t(:,iter) + n;
        Cn = (b(k)/p)*eye(p);
        mu_n = zeros(p,1);
        Mat = zeros(p,p,M);
        tmp = zeros(M,1);
        total = 0;
        for m=1:M
            Mat(:,:,m) = H*Cm(:,:,m)*H' + Cn;
            tmp(m) = alpha(m)*(2*pi)^(-p/2)*(det(Mat(:,:,m)))^(-0.5)*exp(-0.5*(x-(H*mu_m(:,m)+mu_n))'*inv(Mat(:,:,m))*(x-(H*mu_m(:,m)+mu_n))) ;
            total = total + tmp(m);
        end
        t1=0;
        for m = 1:M
            beta_m_X = tmp(m)/total;
            t1 = t1 + beta_m_X*(mu_m(:,m) + Cm(:,:,m)*H'*inv(Mat(:,:,m))*(x-(H*mu_m(:,m)+mu_n)));
        end
        S_hat = t1;
        SE(iter) = norm(t(:,iter)-S_hat)^2;
    end
    optimal_MSE(k) = 10*log10(sum(SE)/Monte_Carlo);
    
    plot(SNR_dB(1:k),optimal_MSE(1:k),'-.rp')
    hold on;grid on;
    plot(SNR_dB(1:k),ssfn_MSE(1:k),'-.bs')
    xlabel('SNR dB');
    ylabel('MSE in dB');
    legend('Optimal','SSFN')
    set(gca,'fontsize',20)
    drawnow
    
end

%% Plot
%close all;
% figure
% set(0,'defaultlinelinewidth',2)
% plot(SNR_dB,optimal_MSE,'-.rs')
% hold on;grid on;
% plot(SNR_dB,ssfn_MSE,'-.bo')
% xlabel('SNR dB');
% ylabel('MSE in dB');
% legend('Optimal estimator','SSFN')
% set(gca,'fontsize',20)

toc;