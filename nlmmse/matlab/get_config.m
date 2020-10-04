%% Gets the correct config setttings
%'r' - Regression config; 'c' - Classification config; 'cd' - Classification denoised config
%'a', 'b', 'c', 'd' - Cases in the experiment
% '_a', '_b', '_c', '_d' - Data type (Shifted mean, Overlapping GMM, Rotated eigen vectors of cov)
% eg. 'rb' - Means experiment 'b' in Regression config
function config = get_config(caseString)
config = [];
switch lower(caseString)
    %
    case "data"
        config.a = logspace(0,3,20); % scaling parameters
        config.len = length(config.a);
        config.b = 50*ones(1,config.len);
        config.sample = 2e3*ones(1,config.len); % number of data points
        config.p = 10*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 50; % No.of simulations for generating ranfom H
        config.file_name = 'data_ring';
        config.folder_name = 'plots/A';
        config.gamma = ones(config.len,config.M);
    
    % Paper experiement data type: A
    case "ra_a"
        config.a = logspace(0,3,20); % scaling parameters
        config.len = length(config.a);
        config.b = 50*ones(1,config.len);
        config.sample = 3e3*ones(1,config.len); % number of data points
        config.p = 10*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_1';
        config.folder_name = 'plots/A';
        config.gamma = ones(config.len,config.M);
        
    case "rb_a_1_a"
        config.b = logspace(3,-4,20);
        config.len = length(config.b);
        config.a = 1*ones(1,config.len); %Do with a=1 and a=10
        config.sample = 3e3*ones(1,config.len);
        config.p = 10*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_2_a_1';
        config.folder_name = 'plots/A';
        config.gamma = ones(config.len,config.M);
        
    case "rb_a_10_a"
        config.b = logspace(3,-4,20);
        config.len = length(config.b);
        config.a = 10*ones(1,config.len); %Do with a=1 and a=10
        config.sample = 3e3*ones(1,config.len);
        config.p = 10*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_2_a_10';
        config.folder_name = 'plots/A';
        config.gamma = ones(config.len,config.M);
        
    case "rc_a"
        config.p = 5:5:60; %dimension of observation x, we are interested in p/q
        config.len = length(config.p);
        config.sample = (1e3/4)*ones(1,config.len);
        config.a = 10*ones(1,config.len);
        config.b = 1*ones(1,config.len);
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_3';
        config.folder_name = 'plots/A';
        config.gamma = ones(config.len,config.M);
      
    case "rd_a"
        config.sample = 1e2:1e3/2:10.1e3;
        config.len = length(config.sample);
        config.a = 5*ones(1,config.len);
        config.b = 1*ones(1,config.len);
        config.p = 10*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_4';
        config.folder_name = 'plots/A';
        config.gamma = ones(config.len,config.M);
        
    case "rc_mismatched_a"      
        config.b_mismatched = logspace(3,-4,20);
        config.len = length(config.b_mismatched);
        config.b = 1*ones(1,config.len);
        config.a = 10*ones(1,config.len); %Do with a=10
        config.sample = 3e3*ones(1,config.len);
        config.p = 10*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_5';
        config.folder_name = 'plots/A';
        config.gamma = ones(config.len,config.M);
        
    case "ca_a"
        config.a = logspace(0,3,20); % scaling parameters
        config.len = length(config.a);
        config.b = 50*ones(1,config.len);
        config.p = 5*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        % This sets size of samples drawn from each Gaussian
        % 1<=n_diff_alpha<=M
        % 1 = Makes all blobs having same alpha and hence same size
        % config.M = Makes all blobs having random alpha an different sizes
        n_diff_alpha = config.M;
        count = randi([1,n_diff_alpha],config.M,1); %mixing proportions
        config.alpha = count/sum(count);
        config.sample = round(3e3*config.alpha); % number of data points converted to nearest integer
        config.Monte_Carlo_H = 20; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_c_1';
        config.folder_name = 'plots/A';
        config.gamma = ones(config.len,config.M);
        
    case "cda_a"
        config.a = logspace(0,3,20); % scaling parameters
        config.len = length(config.a);
        config.b = 50*ones(1,config.len);
        config.p = 5*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        % This sets size of samples drawn from each Gaussian
        % 1<=n_diff_alpha<=M
        % 1 = Makes all blobs having same alpha and hence same size
        % config.M = Makes all blobs having random alpha an different sizes
        n_diff_alpha = config.M;
        count = randi([1,n_diff_alpha],config.M,1); %mixing proportions
        config.alpha = count/sum(count);
        config.sample = round(3e3*config.alpha); % number of data points converted to nearest integer
        config.Monte_Carlo_H = 20; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_c_dn_1';
        config.folder_name = 'plots/A';
        config.gamma = ones(config.len,config.M);
        
    % Paper experiement data type: B
    case "ra_b"
        config.M = 40; % number of Gaussian mixtures
        len = 20;
        gamma = rand(len, config.M);
        for i=1:len
            gamma(i,:) = gamma(i,:)/sum(gamma(i,:)) * i^4;  
        end
        config.gamma = gamma;
        config.len = size(config.gamma,1);
        config.b = 50*ones(1,config.len);
        config.sample = 3e3*ones(1,config.len); % number of data points
        config.p = 10*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.a = zeros(1,config.len); % scaling parameters
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_1';
        config.folder_name = 'plots/B';
        
    case "rb_b"
        config.b = logspace(3,-4,20);
        config.len = length(config.b);      
        config.a = zeros(1,config.len);
        config.sample = 3e3*ones(1,config.len);
        config.p = 10*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_2';
        config.folder_name = 'plots/B';
        config.gamma = 4*ones(config.len,config.M);
        
    case "rc_b"
        config.p = 5:5:60; %dimension of observation x, we are interested in p/q
        config.len = length(config.p);
        config.sample = (1e3/4)*ones(1,config.len);
        config.a = zeros(1,config.len);
        config.b = 1*ones(1,config.len);
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_3';
        config.folder_name = 'plots/B';
        config.gamma = 4*ones(config.len,config.M);
      
    case "rd_b"
        config.sample = 1e2:1e3/2:10.1e3;
        config.len = length(config.sample);
        config.a = zeros(1,config.len);
        config.b = 1*ones(1,config.len);
        config.p = 10*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_4';
        config.folder_name = 'plots/B';
        config.gamma = 4*ones(config.len,config.M);
        
    case "ca_b"
        config.M = 40; % number of Gaussian mixtures
        len = 20;
        gamma = rand(len, config.M);
        for i=1:len
            gamma(i,:) = gamma(i,:)/sum(gamma(i,:)) * i^4;  
        end
        config.gamma = gamma;
        config.len = size(config.gamma,1);
        config.a = zeros(1,config.len); % scaling parameters
        config.b = 50*ones(1,config.len);
        config.p = 5*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        % This sets size of samples drawn from each Gaussian
        % 1<=n_diff_alpha<=M
        % 1 = Makes all blobs having same alpha and hence same size
        % config.M = Makes all blobs having random alpha an different sizes
        n_diff_alpha = 1;
        count = randi([1,n_diff_alpha],config.M,1); %mixing proportions
        config.alpha = count/sum(count);
        config.sample = round(3e3*config.alpha); % number of data points converted to nearest integer
        config.Monte_Carlo_H = 20; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_c_1';
        config.folder_name = 'plots/B';
        
    case "cda_b"
        config.M = 40; % number of Gaussian mixtures
        len = 20;
        gamma = rand(len, config.M);
        for i=1:len
            gamma(i,:) = gamma(i,:)/sum(gamma(i,:)) * i^4;  
        end
        config.gamma = gamma;
        config.len = size(config.gamma,1);
        config.a = zeros(1,config.len); % scaling parameters
        config.b = 50*ones(1,config.len);
        config.p = 5*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        % This sets size of samples drawn from each Gaussian
        % 1<=n_diff_alpha<=M
        % 1 = Makes all blobs having same alpha and hence same size
        % config.M = Makes all blobs having random alpha an different sizes
        n_diff_alpha = 1;
        count = randi([1,n_diff_alpha],config.M,1); %mixing proportions
        config.alpha = count/sum(count);
        config.sample = round(3e3*config.alpha); % number of data points converted to nearest integer
        config.Monte_Carlo_H = 20; % No.of simulations for generating ranfom H
        config.file_name = 'mmse_c_dn_1';
        config.folder_name = 'plots/B';
end
end


