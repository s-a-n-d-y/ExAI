close all;clear;clc;
tic;

%% Plotting properties as latex
set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

experiment = 'a';
switch experiment
    case 'a'
        a = 0:4:100; % scaling parameters
        len = length(a);
        b = 10*ones(1,len);
        sample = 3e3*ones(1,len); % number of data points     
    
    case 'b'
        b = logspace(3,-4,20);
        len = length(b);
        a = 1*ones(1,len);
        sample = 3e3*ones(1,len);
        
    case 'd'
        sample = 1e3:3e3:31e3;
        len = length(sample);
        a = 5*ones(1,len);
        b = 1*ones(1,len);       
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
%% MSE evaluation of SSFN and ELM
ssfn_MSE = zeros(len,1);
elm_MSE = zeros(len,1);
optimal_MSE = zeros(len,1);
for k = 1:len
    mu = a(k)*mu_m; % mean with scaling parameter a(k)
    gm = gmdistribution(mu',Cm,alpha); % Gaussian mixture model
    %% Data generation for SSFN
    
    t = random(gm,sample(k)); % draw random signals from GMM
    t = t';
    H = randn(p,q);
    H = normc(H);
    x = zeros(p,sample(k));
    for i=1:sample(k)
        n = sqrt(b(k)/p)*randn(p,1); %Zero mean Gaussian noise samples
        x(:,i) = H*t(:,i) + n; % noisy signal generation
    end
    x = x'; t = t';
    idx = (randperm(sample(k))<=sample(k)*0.7);
    [ssfn_MSE(k), elm_MSE(k)] = ml_estimator(x(idx,:)',t(idx,:)',x(~idx,:)',t(~idx,:)');
    
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
    
    switch experiment
        case 'a'
            data = SNR_dB(1:k);
            x_label = 'SNR dB';
            
        case 'b'
            data = SNR_dB(1:k);
            x_label = 'SNR dB';
        case 'c'
        case 'd'
            data = sample(1:k);
            x_label = 'Size of dataset';
    end
    
    plot(data,optimal_MSE(1:k),'-.rp')
    hold on;grid on;
    plot(data,ssfn_MSE(1:k),'-.bs')
    hold on;grid on;
    plot(data,elm_MSE(1:k),'-.gs')
    
    xlabel(x_label);
    ylabel('MSE in dB');
    legend('Optimal','SSFN','ELM')
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