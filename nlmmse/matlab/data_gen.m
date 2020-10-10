% Test with following value
% experiment = 'data_ra_a' 'data_rb_a_10';
function data_gen(experiment)
tic;

delete data/*.mat
% rng(10,'twister');
if exist('data/1', 'dir')
   rmdir('data/*','s')
end

%% Plotting properties as latex
set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

%fig = figure('Units','inches',...
%'Position',[0 0 7 4],...
%'PaperPositionMode','auto');

config = get_config(experiment);

a = config.a; % Mean scaling
len = config.len; %No of experiments
b = config.b; % Noise power
sample = config.sample; % number of data points
p = config.p; %dimension of observation x
q = config.q; % Dimension of data t
M = config.M; % number of Gaussian mixtures
Monte_Carlo_H = config.Monte_Carlo_H; % No.of simulations for generating ranfom H
Monte_Carlo_NMSE = config.Monte_Carlo_NMSE; % No.of simulations for evaluating optimal NMSE
gamma = config.gamma;

%% Gaussian mixture model generation
mu = randn(q,M); % Generating random mean vectors
mu = normc(mu); %Normalize columns of mu_m to have unit norm
alpha = (1/M)*ones(M,1); %mixing proportions
T = sum(mu,2);
sig_pow = (q/M).*sum(gamma,2)' + a.^2-(a./M).^2.*trace(T*T');
noise_pow = b;
SNR = (1./noise_pow).*sig_pow; % SNR based on different scaling parameters a
SNR_dB = 10.*log10(SNR);

%% MSE evaluation stats
Cm = zeros(q,q,M,len);
normalized_optimal_MSE = zeros(len,1);

cmap = jet(length(M));  


%%
for k = 1:len
    mkdir("data/"+ k)
    for i=1:M
        Cm(:,:,i,k) = gamma(k,i)*eye(q);
    end
    mu_m = a(k)*mu; % mean with scaling parameter a(k)
    gm = gmdistribution(mu_m',Cm(:,:,:,k),alpha); % Gaussian mixture model
    %% Data generation for experiments
    t = random(gm,sample(k)); % draw random signals from GMM
    
    %colormap with as many colors as groups
    %col_for_val = cmap(group_idx(:), :);
    
    %scatter(t(:,1),t(:,2));
    %hold on;
    
    x = zeros(p(k),sample(k));
    for iter = 1:Monte_Carlo_H   
        H = randn(p(k),q);
        H = normc(H);
        n = sqrt(b(k)/p(k))*randn(p(k),sample(k)); %Zero mean Gaussian noise samples
        x = H*t' + n; % noisy signal generation
        x = x';
        filename = "data/"+ k +"/data_" + iter
        save(filename,'x','t')
    end


    % Optimal NMSE calculator
    t = random(gm,Monte_Carlo_NMSE);
    for iter = 1:Monte_Carlo_NMSE
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
            Mat(:,:,m) = H*Cm(:,:,m,k)*H' + Cn;
            tmp(m) = alpha(m)*(2*pi)^(-p(k)/2)*(det(Mat(:,:,m)))^(-0.5)*exp(-0.5*(x-(H*mu_m(:,m)+mu_n))'*inv(Mat(:,:,m))*(x-(H*mu_m(:,m)+mu_n))) ;
            total = total + tmp(m);
        end
        t_hat=0;
        for m = 1:M
            beta_m_X = tmp(m)/total;
            t_hat = t_hat + beta_m_X*(mu_m(:,m) + Cm(:,:,m,k)*H'*inv(Mat(:,:,m))*(x-(H*mu_m(:,m)+mu_n)));
        end
        SE(iter) = norm(t(iter,:)'-t_hat)^2;
    end
    normalized_optimal_MSE(k) = 10*log10((sum(SE)/Monte_Carlo_NMSE)/sig_pow(k));

end
%%
stats = "data/stats";
normalized_optimal_MSE = normalized_optimal_MSE';
save(stats,'sig_pow','SNR_dB','normalized_optimal_MSE');

toc;
%close all;clear;clc;
