close all;clear;clc;
tic;
experiment = 'a';
switch experiment
    case 'a'
        a = logspace(0,3,20);
        b = 50*ones(1,length(a));
    case 'b'
        b = logspace(2,-3,20);
        a = 5*ones(1,length(b));
end

%% Gaussian mixture model generation
M = 20; % number of Gaussian mixtures
p = 10; %dimension of observation x
q = 10; % Dimension of data t
mu_m1 = randn(q,M); % Generating random mean vectors
mu_m1 = normc(mu_m1); %Normalize columns of mu_m to have unit norm
T = sum(mu_m1,2);
SP = q+a.^2-(a./M).^2.*trace(T*T'); % signal power
SNR = (1./b).*(q+a.^2-(a./M).^2.*trace(T*T')); % SNR based on different scaling parameters a
SNR_dB = 10.*log10(SNR);
Cm = zeros(q,q,M);
for i=1:M
    Cm(:,:,i) = eye(q); %
end
alpha = (1/M)*ones(M,1); %mixing proportions
nSamples = 1000; % No.of simulations for evaluating optimal MSE
H = randn(p,q);
H = normc(H);
%% MSE evaluation of SSFN
ssfn_MSE = zeros(length(SNR),1);
optimal_MSE = zeros(length(SNR),1);
Accuracy = zeros(length(SNR),1);
for k = 1:length(SNR)
    mu_m = a(k)*mu_m1; % mean with scaling parameter a(k)
    
    %% Optimal classifier
    m_true = [];
    %sample_per_class = zeros(M,1);
    for m=1:length(alpha)
        sample_per_class(m) = alpha(m)*nSamples                                                                    ;
        m_true = [m_true; m*ones(sample_per_class(m),1)];
    end
    % Signal Generation
    t = [];
    for m=1:M
        data = mvnrnd(mu_m(:,m)',Cm(:,:,m),sample_per_class(m));
        t = [t, data'];
    end
    m_star = zeros(nSamples,1);
    for iter = 1:nSamples
        H = randn(p,q);
        H = normc(H);
        n = sqrt(b(k)/p)*randn(p,1);
        x = H*t(:,iter) + n; %noisy signal
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
        t_hat = t1;
                        
        for m =1:M
            MU = mu_m(:,m);
            SIGMA = Cm(:,:,m);
            y(m) = alpha(m)*mvnpdf(t_hat,MU,SIGMA);
        end
        [~, m_star(iter)] = max(y);
    end
    %%
    [~,temp] = find(m_star==m_true);
    Accuracy(k) = sum(temp)/nSamples;
    %%
    grid on;
    plot(SNR_dB(1:k),Accuracy(1:k),'-.bp')
    hold on;
    %plot(SNR_dB(1:k),ssfn_MSE(1:k),'-.bs')
    xlabel('SNR dB');
    ylabel('Accuracy');
    legend('Optimal Classifier')
    set(gca,'fontsize',20)
    drawnow
    %%
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
