%% On MMSE Estimation: A Linear Model Under Gaussian Mixture Statistics
close all;clear;clc;
tic;
%% Figure properties
figure(1)
hold on
h = legend('show','location','best');
set(h,'FontSize',12);
grid on;box on;

%% Gaussian mixture model generation
M = 10; % number of Gaussian mixtures
p = 40; %signal dimension
SNR_dB = -20:1:30; % SNR range
SNR = 10.^(SNR_dB./10); % natural scale
Monte_Carlo = 200; % No.of simulations
diff_mu_vals = 10; % No of diff mu to experiment with
cmap=jet(diff_mu_vals);

for mu_val = 1:1:diff_mu_vals
    mu_S = mu_val*mu_val*[1;zeros(p-1,1)];
    for i=2:M
        mu_S(:,i) = circshift(mu_S(:,i-1),1);
    end
    Css = zeros(p,p,M);
    for i=1:M
        C = sqrt(0.01)*randn(p);
        Css(:,:,i) = C'*C; %covariance matrix for each Gaussian
    end
    alpha = (1/M)*ones(M,1); %mixing proportions
    gm = gmdistribution(mu_S',Css,alpha); % Gaussian mixture model
    %% MSE evaluation of MMSE estimator
    MSE = zeros(length(SNR),1);
    for count = 1:length(SNR)
        SE = zeros(Monte_Carlo,1);
        parfor iter = 1:Monte_Carlo
            n = 1; % number of data points
            S = random(gm,n);
            S = S';
            q = 100;
            A = randn(q,p);
            A = normc(A);
            
            signal_power = (1/q)*sum(abs(A*S).^2); %N = number of samples
            noise_power = signal_power/SNR(count);
            W = sqrt(noise_power)*randn(q,1); %Zero mean Gaussian noise samples
            
            X = A*S + W;
            Cw = noise_power*eye(q);
            mu_W = zeros(q,1);
            Mat = zeros(q,q,M);
            tmp = zeros(M,1);
            total = 0;
            for m=1:M
                Mat(:,:,m) = A*Css(:,:,m)*A' + Cw;
                tmp(m) = alpha(m)*(2*pi)^(-p/2)*(det(Mat(:,:,m)))^(-0.5)*exp(-0.5*(X-(A*mu_S(:,m)+mu_W))'*inv(Mat(:,:,m))*(X-(A*mu_S(:,m)+mu_W))) ;
                total = total + tmp(m);
            end
            t1=0;
            for m = 1:M
                beta_m_X = tmp(m)/total;
                t1 = t1 + beta_m_X*(mu_S(:,m) + Css(:,:,m)*A'*inv(Mat(:,:,m))*(X-(A*mu_S(:,m)+mu_W)));
            end
            S_hat = t1;
            
            SE(iter) = norm(S-S_hat)^2;
            
        end
        MSE(count) = 10*log10(sum(SE)/Monte_Carlo);
    end
    figure(1), plot(SNR_dB(1:count),MSE(1:count),'-ro','markers',4, 'color',cmap(mu_val,:),'DisplayName',strcat('Mean Scaling factor = ', num2str(mu_val*mu_val)));
    drawnow
end
%% Plot
xlabel('SNR dB');
ylabel('MSE in dB');
set(gca,'fontsize',20)

toc;
