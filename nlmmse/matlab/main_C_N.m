% Test with following value
% experiment = 'ca';
function main_C_N(experiment)
tic;

%% Plotting properties as latex
set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

fig = figure('Units','inches',...
'Position',[0 0 7 4],...
'PaperPositionMode','auto');

config = get_config(experiment);

a = config.a; % Mean scaling
len = config.len; %No of experiments
b = config.b; % Noise power
p = config.p; %dimension of observation x
q = config.q; % Dimension of data t
M = config.M; % number of Gaussian mixtures
alpha = config.alpha; % Mixing proportion probability
sample = config.sample; % number of data points in each mixture
Monte_Carlo_H = config.Monte_Carlo_H; % No.of simulations for generating ranfom H


%% Gaussian mixture model generation
mu = randn(q,M); % Generating random mean vectors
mu = normc(mu); %Normalize columns of mu_m to have unit norm
T = sum(mu,2);
sig_pow = (q+a.^2-(a./M).^2.*trace(T*T'));
noise_pow = b;
SNR = (1./noise_pow).*sig_pow; % SNR based on different scaling parameters 'a'
SNR_dB = 10.*log10(SNR);
Cm = zeros(q,q,M);
%t = zeros(sum(sample),q);
for i=1:M
    %C = sqrt(0.01)*randn(p);
    %Css(:,:,i) = C'*C; %covariance matrix for each Gaussian
    Cm(:,:,i) = eye(q); %
end

%% CE evaluation of optimal estimator
mean_CE = zeros(len,1);
mean_ssfn_CE = zeros(len,1);
mean_elm_CE = zeros(len,1);

for k = 1:len
    %% Sample data generation
    n_samples = sum(sample);
    m_star = zeros(n_samples,1);
    mu_m = a(k)*mu; % mean with scaling parameter a(k)
    m_true = [];
    t = [];
    for m=1:M
        data = mvnrnd(mu_m(:,m)', Cm(:,:,m), sample(m));
        m_true = [m_true; m*ones(sample(m),1)];
        t = [t, data'];
    end
    
    onehot = @(X)bsxfun(@eq, X(:), 1:max(X));
    m_true_onehot = double(onehot(m_true))';
    
    %% Evaluation for SSFN and ELM
    parfor iter = 1:Monte_Carlo_H
        y = zeros(M,1);
        H = randn(p(k),q);
        H = normc(H);        
        n = sqrt(b(k)/p(k))*randn(p(k),n_samples); %Zero mean Gaussian noise samples
        x = H*t + n; % noisy signal generation        
        x = x';
        idx = (randperm(n_samples)<=n_samples*0.7);
        [~, ~, ssfn_acc(iter), elm_acc(iter)] = ml_estimator(x(idx,:)',m_true_onehot(:,idx),x(~idx,:)',m_true_onehot(:,~idx));
    end
    mean_ssfn_CE(k) = sum(ssfn_acc)/Monte_Carlo_H;
    mean_elm_CE(k) = sum(elm_acc)/Monte_Carlo_H;
    
    %% Monte Carlo for H, by drawing one sample at a time
    parfor iter = 1:n_samples
        y = zeros(M,1);
        H = randn(p(k),q);
        H = normc(H);
        n = sqrt(b(k)/p(k))*randn(p(k),1);
        x = H*t(:,iter) + n; %noisy signal
        Cn = (b(k)/p(k))*eye(p(k));
        mu_n = zeros(p(k),1);
        for m =1:M
            MU = [H, eye(p(k))]*[mu_m(:,m); mu_n];
            SIGMA = [H, eye(p(k))]*[Cm(:,:,m), zeros(q,p(k));zeros(p(k),q), Cn]*[H, eye(p(k))]';
            y(m) = alpha(m)*mvnpdf(x,MU,SIGMA);
        end
        [~, m_star(iter)] = max(y);
    end
    
    [~,count] = find(m_star==m_true);
    mean_CE(k) = sum(count)/n_samples;
     
    switch experiment
        case 'ca'
            data = SNR_dB(1:k);
            x_label = 'SNR (dB)';
            file_name = 'mmse_c_1';
            xlim([-10 35])
            plot_title = {['P = ' num2str(p(k)) ', Q = ' num2str(q)]
                          ['b = ' num2str(b(k))]};
            
        case 'cb'
            data = SNR_dB(1:k);
            x_label = 'SNR (dB)';
            file_name = 'mmse_c_2';
            plot_title = {['SNR = ' num2str(SNR_dB(k)) ', P = ' num2str(p(k)) ', Q = ' num2str(q)]
                          ['a = ' num2str(a(k)) ' and b = ' num2str(b(k))]};
            
        case 'cc'
            data = p(1:k);
            x_label = 'Dimension of observation (P) w.r.t. a given Dimension of data (Q=10)';
            file_name = 'mmse_c_3';
            plot_title = {['SNR = ' num2str(SNR_dB(k)) ', P = ' num2str(p(k)) ', Q = ' num2str(q)]
                          ['a = ' num2str(a(k)) ' and b = ' num2str(b(k))]};
            
        case 'cd'
            data = sample(1:k);
            x_label = 'Size of dataset';
            file_name = 'mmse_c_4';
            plot_title = {['SNR = ' num2str(SNR_dB(k)) ', P = ' num2str(p(k)) ', Q = ' num2str(q)]
                          ['a = ' num2str(a(k)) ' and b = ' num2str(b(k))]};
    end
    
    plot(data,mean_CE(1:k),'-.rp','MarkerSize',2)
    hold on;grid on;
    plot(data,mean_ssfn_CE(1:k),'-.bs','MarkerSize',2)
    hold on;grid on;
    plot(data,mean_elm_CE(1:k),'-.go','MarkerSize',2)
    legend_label = {'Optimal' 'SSFN' 'ELM'};
    y_label = 'Accuracy';
    set_plot_property(fig, x_label, y_label, legend_label, plot_title, file_name);
    
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
%close all;clear;clc;