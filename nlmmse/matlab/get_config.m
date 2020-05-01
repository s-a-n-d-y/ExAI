%% Gets the correct config setttings
%'r' - Regression config; 'c' - Classification config; 'cd' - Classification denoised config
%'a', 'b', 'c', 'd' - Cases in the experiment
% eg. 'rb' - Means experiment 'b' in Regression config
function config = get_config(caseString)
config = [];
switch lower(caseString)
    case "ra"
        config.a = logspace(0,3,20); % scaling parameters
        config.len = length(config.a);
        config.b = 50*ones(1,config.len);
        config.sample = 3e3*ones(1,config.len); % number of data points
        config.p = 10*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        
    case "rb_a_1"
        config.b = logspace(3,-4,20);
        config.len = length(config.b);
        config.a = 10*ones(1,config.len); %Do with a=1 and a=10
        config.sample = 3e3*ones(1,config.len);
        config.p = 10*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        
    case "rb_a_10"
        config.b = logspace(3,-4,20);
        config.len = length(config.b);
        config.a = 10*ones(1,config.len); %Do with a=1 and a=10
        config.sample = 3e3*ones(1,config.len);
        config.p = 10*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        
    case "rc"
        config.p = 5:5:60; %dimension of observation x, we are interested in p/q
        config.len = length(config.p);
        config.sample = (1e3/4)*ones(1,config.len);
        config.a = 5*ones(1,config.len);
        config.b = 1*ones(1,config.len);
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        
    case "rd"
        config.sample = 1e2:1e3/2:10.1e3;
        config.len = length(config.sample);
        config.a = 5*ones(1,config.len);
        config.b = 1*ones(1,config.len);
        config.p = 10*ones(1,config.len); %dimension of observation x
        config.q = 10; % Dimension of data t
        config.M = 40; % number of Gaussian mixtures
        config.Monte_Carlo_NMSE = 1000; % No.of simulations for evaluating optimal MSE
        config.Monte_Carlo_H = 100; % No.of simulations for generating ranfom H
        
    case "ca"
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
        n_diff_alpha = 1;
        count = randi([1,n_diff_alpha],config.M,1); %mixing proportions
        config.alpha = count/sum(count);
        config.sample = round(3e3*config.alpha); % number of data points converted to nearest integer
        config.Monte_Carlo_H = 20; % No.of simulations for generating ranfom H
        
    case "cda"
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
        n_diff_alpha = 1;
        count = randi([1,n_diff_alpha],config.M,1); %mixing proportions
        config.alpha = count/sum(count);
        config.sample = round(3e3*config.alpha); % number of data points converted to nearest integer
        config.Monte_Carlo_H = 20; % No.of simulations for generating ranfom H
end
end


