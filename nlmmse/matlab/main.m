close all;clear;clc;
tic;

%% Plotting properties as latex
set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

experiment = 'd';
switch experiment
    case 'a'
        a = logspace(0,3,20); % scaling parameters
        len = length(a);
        b = 50*ones(1,len);
        sample = 3e3*ones(1,len); % number of data points
        p = 10*ones(1,len); %dimension of observation x
        q = 10; % Dimension of data t
        M = 40; % number of Gaussian mixtures
    
    case 'b'
        b = logspace(3,-4,20);
        len = length(b);
        a = 1*ones(1,len);
        sample = 3e3*ones(1,len);
        p = 10*ones(1,len); %dimension of observation x
        q = 10; % Dimension of data t
        M = 40; % number of Gaussian mixtures
        
    case 'c'
        p = 5:5:60; %dimension of observation x, we are interested in p/q
        len = length(p);
        sample = (1e3/4)*ones(1,len);
        a = 5*ones(1,len);
        b = 1*ones(1,len);
        q = 20; % Dimension of data t
        M = 40; % number of Gaussian mixtures
        
    case 'd'
        sample = 1e2:1e3/2:10.1e3;
        len = length(sample);
        a = 5*ones(1,len);
        b = 1*ones(1,len);
        p = 10*ones(1,len); %dimension of observation x
        q = 10; % Dimension of data t
        M = 40; % number of Gaussian mixtures
end

%% Gaussian mixture model generation
mu = randn(q,M); % Generating random mean vectors
mu = normc(mu); %Normalize columns of mu_m to have unit norm
T = sum(mu,2);
SNR = (1./b).*(q+a.^2-(a./M).^2.*trace(T*T')); % SNR based on different scaling parameters a
SNR_dB = 10.*log10(SNR);
Cm = zeros(q,q,M);
for i=1:M
    %C = sqrt(0.01)*randn(p);
    %Css(:,:,i) = C'*C; %covariance matrix for each Gaussian
    Cm(:,:,i) = eye(q); %
end
alpha = (1/M)*ones(M,1); %mixing proportions
Monte_Carlo_MMSE = 1000; % No.of simulations for evaluating optimal MSE
Monte_Carlo_H = 100; % No.of simulations for generating ranfom H

%% MSE evaluation of SSFN and ELM
ssfn_MSE = zeros(len,1);
elm_MSE = zeros(len,1);
optimal_MSE = zeros(len,1);
for k = 1:len
    mu_m = a(k)*mu; % mean with scaling parameter a(k)
    gm = gmdistribution(mu_m',Cm,alpha); % Gaussian mixture model
    %% Data generation for SSFN and ELM
    t = random(gm,sample(k)); % draw random signals from GMM
    parfor iter = 1:Monte_Carlo_H   
        H = randn(p(k),q);
        H = normc(H);
        x = zeros(p(k),sample(k));
        for i=1:sample(k)
            n = sqrt(b(k)/p(k))*randn(p(k),1); %Zero mean Gaussian noise samples
            x(:,i) = H*t(i,:)' + n; % noisy signal generation
        end
        x = x';
        idx = (randperm(sample(k))<=sample(k)*0.7);
        [ssfn_SE(iter), elm_SE(iter)] = ml_estimator(x(idx,:)',t(idx,:)',x(~idx,:)',t(~idx,:)');
    end
    ssfn_MSE(k) = 10*log10(sum(ssfn_SE)/Monte_Carlo_H);
    elm_MSE(k) = 10*log10(sum(elm_SE)/Monte_Carlo_H);
    %% MSE evaluation of MMSE estimator
    
    t = random(gm,Monte_Carlo_MMSE);
    parfor iter = 1:Monte_Carlo_MMSE
        H = randn(p(k),q);
        H = normc(H);
        n = sqrt(b(k)/p(k))*randn(p(k),1); %Zero mean Gaussian noise samples
        
        x = H*t(iter,:)' + n;
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
        t1=0;
        for m = 1:M
            beta_m_X = tmp(m)/total;
            t1 = t1 + beta_m_X*(mu_m(:,m) + Cm(:,:,m)*H'*inv(Mat(:,:,m))*(x-(H*mu_m(:,m)+mu_n)));
        end
        S_hat = t1;
        SE(iter) = norm(t(iter,:)'-S_hat)^2;
    end
    optimal_MSE(k) = 10*log10(sum(SE)/Monte_Carlo_MMSE);
    
    switch experiment
        case 'a'
            data = SNR_dB(1:k);
            x_label = 'SNR dB';
            
        case 'b'
            data = SNR_dB(1:k);
            x_label = 'SNR dB';
            
        case 'c'
            data = p(1:k);
            x_label = 'Dimension of observation (P) w.r.t. a given Dimension of data (Q=10)';
            
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