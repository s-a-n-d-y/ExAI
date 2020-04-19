%figure(fig2), plot(SNR_dB(1:count),MNME(1:count),'-ro','markers',4, 'color',cmap(mu_val,:),'DisplayName',strcat('Mean Scaling factor = ', num2str(mu_val*mu_val)));
close all;clear;clc;

%% Figure properties
figure(1)
hold on
h = legend('show','location','best');
set(h,'FontSize',12);
grid on;box on;

%% Gaussian mixture model generation
M = 10; % number of Gaussian mixtures
p = 40; %signal dimension
q = 100; %signal feature space dimension
n = 1e3; %no of samples
Monte_Carlo = 50; % No.of simulations
diff_mu_vals = 10; % No of diff mu to experiment with
SNR_dB = -20:1:40; % SNR range
SNR = 10.^(SNR_dB./10); % natural scale
cmap=jet(diff_mu_vals);

%% Data generation
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
    ssfn_MSE = zeros(length(SNR),1);
    for count = 1:length(SNR)
        ssfn_SE = zeros(Monte_Carlo,1);
        parfor iter = 1:Monte_Carlo
            S = random(gm,n); % draw a random signal from GMM
            S = S';
            A = randn(q,p);
            A = normc(A);
            X = zeros(q,n);
            for i=1:n
                signal_power = (1/q)*sum(abs(A*S(:,i)).^2); %N = number of samples
                noise_power = signal_power/SNR(count);
                W = sqrt(noise_power)*randn(q,1); %Zero mean Gaussian noise samples
                X(:,i) = A*S(:,i) + W; % noisy signal generation
            end
            X = X'; S = S';
            idx = (randperm(n)<=n*0.7);
            [~, ssfn_SE(iter)] = ssfn_estimator(X(idx,:)',S(idx,:)',X(~idx,:)',S(~idx,:)');
        end
        ssfn_MSE(count) = sum(ssfn_SE)/Monte_Carlo;
    end
    figure(1), plot(SNR_dB(1:count),ssfn_MSE(1:count),'-ro','markers',4, 'color',cmap(mu_val,:),'DisplayName',strcat('Mean Scaling factor = ', num2str(mu_val*mu_val)));
end

%% Plot
xlabel('SNR dB');
ylabel('NME in dB');
set(gca,'fontsize',20)
