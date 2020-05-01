close all;clear;clc;
tic;

%% Plotting properties as latex
set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

experiment = 'ra';
config = get_config(experiment);

a = config.a; % Mean scaling
len = config.len; %No of experiments
b = config.b; % Noise power
sample = config.sample; % number of data points
p = config.p; %dimension of observation x
q = config.q; % Dimension of data t
M = config.M; % number of Gaussian mixtures
Monte_Carlo_NMSE = config.Monte_Carlo_NMSE; % No.of simulations for evaluating optimal NMSE
Monte_Carlo_H = config.Monte_Carlo_H; % No.of simulations for generating ranfom H


%% Gaussian mixture model generation
mu = randn(q,M); % Generating random mean vectors
mu = normc(mu); %Normalize columns of mu_m to have unit norm
alpha = (1/M)*ones(M,1); %mixing proportions
T = sum(mu,2);
sig_pow = (q+a.^2-(a./M).^2.*trace(T*T'));
noise_pow = b;
SNR = (1./noise_pow).*sig_pow; % SNR based on different scaling parameters a
SNR_dB = 10.*log10(SNR);
Cm = zeros(q,q,M);
for i=1:M
    %C = sqrt(0.01)*randn(p);
    %Css(:,:,i) = C'*C; %covariance matrix for each Gaussian
    Cm(:,:,i) = eye(q); %
end

%% MSE evaluation of SSFN and ELM
ssfn_normalized_MSE = zeros(len,1);
elm_normalized_MSE = zeros(len,1);
normalized_MSE = zeros(len,1);
for k = 1:len
    mu_m = a(k)*mu; % mean with scaling parameter a(k)
    gm = gmdistribution(mu_m',Cm,alpha); % Gaussian mixture model
    %% Data generation for SSFN and ELM
    t = random(gm,sample(k)); % draw random signals from GMM
    x = zeros(p(k),sample(k));
    parfor iter = 1:Monte_Carlo_H   
        H = randn(p(k),q);
        H = normc(H);
        n = sqrt(b(k)/p(k))*randn(p(k),sample(k)); %Zero mean Gaussian noise samples
        x = H*t' + n; % noisy signal generation
        x = x';
        idx = (randperm(sample(k))<=sample(k)*0.7);
        [ssfn_SE(iter), elm_SE(iter), ~, ~] = ml_estimator(x(idx,:)',t(idx,:)',x(~idx,:)',t(~idx,:)');
    end
    ssfn_normalized_MSE(k) = 10*log10((sum(ssfn_SE)/Monte_Carlo_H)/sig_pow(k));
    elm_normalized_MSE(k) = 10*log10((sum(elm_SE)/Monte_Carlo_H)/sig_pow(k));
    %% MSE evaluation of MMSE estimator
    
    t = random(gm,Monte_Carlo_NMSE);
    parfor iter = 1:Monte_Carlo_NMSE
        H = randn(p(k),q);
        H = normc(H);
        n = sqrt(b(k)/p(k))*randn(p(k),1); %Zero mean Gaussian noise samples  
        x = H*t(iter,:)' + n; %Observation generation
        Cn = (b(k)/p(k))*eye(p(k));
        mu_n = zeros(p(k),1);
        Mat = zeros(p(k),p(k),M);
        tmp = zeros(M,1);
        total = 0;
        for m=1:M
            Mat(:,:,m) = H*Cm(:,:,m)*H' + Cn;
            tmp(m) = alpha(m)*(2*pi)^(-p(k)/2)*(det(Mat(:,:,m)))^(-0.5)*exp(-0.5*(x-(H*mu_m(:,m)+mu_n))'*inv(Mat(:,:,m))*(x-(H*mu_m(:,m)+mu_n))) ;
            total = total + tmp(m);
        end
        t_hat=0;
        for m = 1:M
            beta_m_X = tmp(m)/total;
            t_hat = t_hat + beta_m_X*(mu_m(:,m) + Cm(:,:,m)*H'*inv(Mat(:,:,m))*(x-(H*mu_m(:,m)+mu_n)));
        end
        SE(iter) = norm(t(iter,:)'-t_hat)^2;
    end
    normalized_MSE(k) = 10*log10((sum(SE)/Monte_Carlo_NMSE)/sig_pow(k));
    
    switch experiment
        case 'ra'
            data = SNR_dB(1:k);
            x_label = 'SNR (dB)';
            
        case 'rb'
            data = SNR_dB(1:k);
            x_label = 'SNR (dB)';
            
        case 'rc'
            data = p(1:k);
            x_label = 'Dimension of observation (P) w.r.t. a given Dimension of data (Q=10)';
            
        case 'rd'
            data = sample(1:k);
            x_label = 'Size of dataset';
    end
    
    plot(data,normalized_MSE(1:k),'-.rp')
    hold on;grid on;
    plot(data,ssfn_normalized_MSE(1:k),'-.bs')
    hold on;grid on;
    plot(data,elm_normalized_MSE(1:k),'-.gs')
    
    xlabel(x_label);
    ylabel('NMSE (dB)');
    legend('Optimal','SSFN','ELM')
    set(gca,'fontsize',20)
    
    title({
    ['SNR = ' num2str(SNR_dB(k)) ', P = ' num2str(p(k)) ', Q = ' num2str(q)] 
    ['a = ' num2str(a(k)) ' and b = ' num2str(b(k))]
    });
    
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