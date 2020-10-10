% Test with following value
% experiment = 'ra';
function data_vis(experiment)
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
cmap = jet(length(M));  

%%
stats = "data/stats"
save(stats,'sig_pow','SNR_dB')

exp_1 = "data/exp_1"
x_snr = [-6.58249031200274,-6.18561361846295,-5.46420286771958,-4.26941059762383,-2.51529842475687,-0.242729299331180,2.41256942348224,5.30499912218031,8.32853718806938,11.4199775780973,14.5453872217894,17.6874946158789,20.8377388206352,23.9919312724467,27.1480355888674,30.3050647633731,33.4625411181353,36.6202336393892,39.7780306434733,42.9358781459669];
y_optimal = [-0.619100624177964,-0.660455458691819,-0.738833490150500,-0.927251287995877,-1.33579854464517,-1.95002530816507,-2.91118505358461,-3.91188779331992,-5.22512355301361,-6.56658199261872,-7.71967442550900,-8.97576905130873,-9.95193764688905,-10.6637216722364,-11.2941207277486,-11.8136737272375,-11.8739490676789,-11.8036377413611,-12.2655178311187,-12.1134818119713];
y_ssfn = [-0.447975244573882,-0.499306841155385,-0.589589769909751,-0.820364830854633,-1.31672862437220,-2.14548883923237,-3.62913695826874,-5.54590634951198,-8.21595073878540,-11.0022020443288,-13.4752908328947,-15.7896286927383,-18.1592416154551,-20.0893501599424,-22.0239176099921,-23.8475428392465,-25.1635342636534,-26.8697515270431,-28.5545710845006,-30.1641197274040];
y_elm = [-0.897646891218553,-0.798645168844431,-0.887019050102097,-1.16522646198850,-1.80291539931001,-2.85327573617060,-4.82438885295286,-8.79923171955261,-14.4293792441198,-18.9060574934210,-22.0991276657209,-25.3707300726165,-28.4265092008580,-31.5909492933880,-34.7534849016691,-37.8812656842734,-41.0915363164839,-44.2771465957668,-47.3916514765553,-50.5628571796373];
y_cnn = [-0.8093369677221595, -0.8350562633489471, -0.990662640930214, -1.1277862245381305, -1.6064183299041377, -2.2314771751661553, -3.412060976774907, -4.583274375238893, -5.929830113373448, -6.7170923451970355, -7.649530423322699, -8.222206699287248, -8.38889304865448, -9.106327180349869, -8.240292740557123, -8.780180913645898, -8.786221497036825, -9.06424533603416, -8.189694021863152, -8.01397268020858];
save(exp_1, 'x_snr', 'y_optimal', 'y_ssfn', 'y_elm', 'y_cnn');
% 
% xlim([-8 20])
% ylim([-28 5])
% hold on;grid on;
% plot(x_snr, y_elm,'-.rp','MarkerSize',2)
% hold on;grid on;
% plot(x_snr, y_ssfn,'-.bs','MarkerSize',2)
% hold on;grid on;
% plot(x_snr, y_optimal,'-.gs','MarkerSize',2)
% hold on;grid on;
% plot(x_snr, y_cnn,'-.cs','MarkerSize',2)


for k = 1:len
    mkdir("data/"+ k)
    for i=1:M
        Cm(:,:,i,k) = gamma(k,i)*eye(q);
    end
    mu_m = a(k)*mu; % mean with scaling parameter a(k)
    gm = gmdistribution(mu_m',Cm(:,:,:,k),alpha); % Gaussian mixture model
    %% Data generation for SSFN and ELM
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
