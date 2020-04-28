
function [ssfn_mean_test_err, elm_mean_test_err] = ml_estimator(X_train,T_train,X_test,T_test)
addpath(genpath('ssfn/Datasets'), genpath('ssfn/Functions'), genpath('elm'));


a_leaky_RLU=0;      %   set to a small non-zero value if you want to test leaky-RLU
g=@(x) x.*(x >= 0)+a_leaky_RLU*x.*(x < 0);


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % %       SSFN      % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

lam=1e2;
mu=1e3;
kmax=100;
alpha=2;
nmax=10;
eta_n=0.005; %-Inf
%eta_l=0.1; %-Inf -> If the improvement is less than 0.1 then stop
eta_l=-Inf;
lmax=20;
% Delta=50; % If nmax = Delta then no of nodes are fixed
Delta=nmax;

%   Loading the dataset
%[X_train,T_train,X_test,T_test]=Load_dataset(Database_name);

Q=size(T_train,1);  %   Target Dimension
trialNum=10;

% The perfomance measures we are interested in
train_err_SSFN=zeros(1,trialNum);
test_err_SSFN=zeros(1,trialNum);

train_err_ELM=zeros(1,trialNum);
test_err_ELM=zeros(1,trialNum);

%   Generating the set of nodes in each layer
NumNodes_min=2*Q;
NumNodes_max=2*Q+nmax;
temp=NumNodes_min:Delta:NumNodes_max;
ind=ones(lmax,1);
NumNodes=temp(ind,:);

eps_o=alpha*sqrt(2*Q);  %   the regularization constant
First_Block='LS';

% % Finding the optimum number of random nodes in each layer
[~, ~, ~, ~, ~, NumNodes_opt]=SSFN(X_train, T_train,...
    X_test, T_test, g, NumNodes, eps_o, mu, kmax, lam, eta_n, eta_l, First_Block);

% Running the network with the optimum number nodes in each layer derived above
% disp(NumNodes_opt);
parfor i=1:trialNum
    % Loading the dataset each time to reduce the effect of random partitioning in some of the datasets
    % [X_train,T_train,X_test,T_test]=Load_dataset(Database_name);
    
    [train_error_ssfn, test_error_ssfn, ~, ~, ~, ~]=SSFN(X_train, T_train,...
        X_test,T_test, g, NumNodes_opt', eps_o, mu, kmax, lam, eta_n, eta_l, First_Block);
    
    train_err_SSFN(i) = train_error_ssfn(end);
    test_err_SSFN(i) = test_error_ssfn(end);
    
    % ELM estimator
    [~, ~, train_error_elm, test_error_elm]=ELM(X_train, T_train, X_test, T_test, lam, (2*Q+nmax));
    train_err_ELM(i) = train_error_elm(end);
    test_err_ELM(i) = test_error_elm(end);

end

ssfn_mean_train_err = mean(train_err_SSFN);
ssfn_mean_test_err = mean(test_err_SSFN);

elm_mean_train_err = mean(train_err_ELM);
elm_mean_test_err = mean(test_err_ELM);

disp([ 'Train and Test estimate error SSFN = ',num2str(ssfn_mean_train_err),' , ',num2str(ssfn_mean_test_err)]);
disp([ 'Train and Test estimate error ELM = ',num2str(elm_mean_train_err),' , ',num2str(elm_mean_test_err)])

