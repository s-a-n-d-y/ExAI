%%  Name:   SSFN_Performance
%
%   Generating the performance results of SSFN shown in Table 2 and Table 4
%
%   Data:   Benchmark datasets mentioned in the paper
%
%   Output: Mean and standard deviation and testing accuracy over multiple
%           trials of SSFN for classification
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Paper:              SSFN: Low Complexity Self Size-estimating Feed-forward Neural Network using Layer-wise Convex Optimization
%   Authors:          Saikat Chatterjee, Alireza M. Javid, Mostafa Sadeghi, Shumpei Kikuta, Partha P. Mitra, Mikael Skoglund
%   Organiztion:    KTH Royal Institute of Technology
%   Contact:          Saikat Chatterjee (sach@kth.se), Alireza Javid (almj@kth.se) 
%   Website:         www.ee.kth.se/reproducible/
%
%   ***April 2019***

%% begining of the simulation


addpath(genpath('Datasets'));
addpath(genpath('Functions'));

a_leaky_RLU=0;      %   set to a small non-zero value if you want to test leaky-RLU
g=@(x) x.*(x >= 0)+a_leaky_RLU*x.*(x < 0);

%%  Choosing a dataset
% Choose one of the following datasets:

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % %       SSFN      % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

Database_name='Vowel';                               lam=1e2;        mu=1e3;        kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;      lmax=20;        Delta=50;
% Database_name='Satimage';                          lam=1e6;        mu=1e5;        kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;      lmax=20;        Delta=50;
% Database_name='Caltech101';                       lam=5e0;        mu=1e-2;       kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;      lmax=20;        Delta=50;
% Database_name='Letter';                               lam=1e-5;       mu=1e4;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;      lmax=20;        Delta=50;
% Database_name='NORB';                               lam=1e2;        mu=1e2;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;      lmax=20;        Delta=50;
% Database_name='Shuttle';                             lam=1e5;        mu=1e4;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;      lmax=20;        Delta=50;
% Database_name='MNIST';                              lam=1e0;        mu=1e5;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;      lmax=20;        Delta=50;
% Database_name='CIFAR-10';                          lam=1e8;        mu=1e3;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;      lmax=20;        Delta=50;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % %       hSSFN      % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% Database_name='Vowel';                                lam=1e2;        mu=1e3;         kmax=100;       alpha=2;        nmax=4000;        eta_n=0.005;         eta_l=0.05;       lmax=20;        Delta=500;
% Database_name='Satimage';                           lam=1e6;        mu=1e5;         kmax=100;       alpha=2;        nmax=4000;        eta_n=0.005;         eta_l=0.15;       lmax=20;        Delta=500;
% Database_name='Caltech101';                        lam=5e0;        mu=1e-2;        kmax=100;       alpha=3;        nmax=20;            eta_n=0.005;         eta_l=0.15;       lmax=20;        Delta=5;
% Database_name='Letter';                                lam=1e-5;       mu=1e4;         kmax=100;       alpha=2;         nmax=4000;        eta_n=0.005;         eta_l=0.25;       lmax=20;        Delta=500;
% Database_name='NORB';                                 lam=1e2;        mu=1e2;         kmax=100;       alpha=2;        nmax=4000;        eta_n=0.005;         eta_l=0.15;       lmax=20;        Delta=500;
% Database_name='Shuttle';                              lam=1e5;        mu=1e4;         kmax=100;       alpha=2;         nmax=4000;        eta_n=0.005;         eta_l=0.05;       lmax=20;        Delta=500;
% Database_name='MNIST';                               lam=1e0;        mu=1e5;         kmax=100;       alpha=2;         nmax=4000;        eta_n=0.005;         eta_l=0.15;       lmax=20;        Delta=500;
% Database_name='CIFAR-10';                           lam=1e8;        mu=1e3;         kmax=100;       alpha=2;         nmax=4000;        eta_n=0.005;         eta_l=0.15;       lmax=20;        Delta=500;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

%   Loading the dataset
[X_train,T_train,X_test,T_test]=Load_dataset(Database_name);

Q=size(T_train,1);  %   Target Dimension
trialNum=50;

% The perfomance measures we are interested in
accuracy_SSFN=zeros(1,trialNum);

%   Generating the set of nodes in each layer
NumNodes_min=2*Q;
NumNodes_max=2*Q+1000;
temp=NumNodes_min:Delta:NumNodes_max;
ind=ones(lmax,1);
NumNodes=temp(ind,:);

eps_o=alpha*sqrt(2*Q);  %   the regularization constant
First_Block='LS';

% % Finding the optimum number of random nodes in each layer
[~, ~, train_accuracy, test_accuracy, ~, NumNodes_opt]=SSFN(X_train, T_train,...
    X_test, T_test, g, NumNodes, eps_o, mu, kmax, lam, eta_n, eta_l, First_Block);

% Running the network with the optimum number nodes in each layer derived above

for i=1:trialNum
            %   Loading the dataset each time to reduce the effect of random partitioning in some of the datasets
            [X_train,T_train,X_test,T_test]=Load_dataset(Database_name);

            [~,~,train_accuracy,test_accuracy,~,~]=SSFN(X_train, T_train,...
                X_test,T_test, g, NumNodes_opt', eps_o, mu, kmax, lam, eta_n, eta_l, First_Block);
            
            accuracy_SSFN(i)=test_accuracy(end);
end

mean_accuracy=mean(accuracy_SSFN);
std_accuracy=std(accuracy_SSFN);

% Displaying the results of SSFN
disp(['Performance results of "',Database_name,'" dataset:'])
disp([ 'Test accuracy = ',num2str(100*mean_accuracy),'+',num2str(100*std_accuracy)])

