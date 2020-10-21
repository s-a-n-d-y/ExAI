%%  Name:   SSFN_Behavior
%
%   Generating the performance results of SSFN shown in Figure 3
%
%   Data:   Benchmark datasets mentioned in the paper
%
%   Output: Train and Test classification accuracy versus the total number
%   of random nodes in the network
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

clc; clear variables; clear global; close all;

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

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

%   Loading the dataset
[X_train,T_train,X_test,T_test]=Load_dataset(Database_name);

P=size(X_train,1);  %   Data Dimension
Q=size(T_train,1);  %   Target Dimension

%   Generating the set of nodes in each layer
NumNodes_min=2*Q;
NumNodes_max=2*Q+1000;
temp=NumNodes_min:Delta:NumNodes_max;
ind=ones(lmax,1);
NumNodes=temp(ind,:);

eps_o=alpha*sqrt(2*Q);  %   the regularization constant
First_Block='LS';

VQ=[eye(Q);-eye(Q)];

size_counter=0;
train_error=[];
test_error=[];
test_accuracy=[];
train_accuracy=[];
Total_NN_size=[];   %   the sequence of total number of random nodes in the network
NumNode_opt=[];     %   The set of optimum number of nodes in each layer

%% First layer Block
switch First_Block
    case {'LS'}
        [train_label_firstBlock,test_label_firstBlock,train_accuracy_firstBlock,test_accuracy_firstBlock]=LS(...
            X_train,T_train,X_test,T_test, lam);
end

train_error(1)=Calculate_error(T_train,train_label_firstBlock);
test_error(1)=Calculate_error(T_test,test_label_firstBlock);
test_accuracy(1)=test_accuracy_firstBlock;
train_accuracy(1)=train_accuracy_firstBlock;
Total_NN_size=[Total_NN_size,size_counter];     %   At this point, the total number of random nodes is zero in the network

%   Initializing the algorithm for the first time
Yi=X_train;
t_hati=train_label_firstBlock;
Pi=P;

%%%%%%%%    Test
Yi_test=X_test;
t_hati_test=test_label_firstBlock;
%%%%%%%%

Thr_l=1;    %   The flag correspoding to eta_l
layer=0;
while layer<size(NumNodes,1)
    layer=layer+1;
    
    if Thr_l==1
        Ri=2*rand(NumNodes(layer,1)-2*Q, Pi)-1;     %   Generating the random matrix R
        Zi_part1=VQ*t_hati;
        Zi_part1_test=VQ*t_hati_test;
        
        Thr_n=1;    %   The flag correspoding to eta_n
        i=0;
        while i<size(NumNodes,2)
            i=i+1;
            if i==2
                Thr_n=1;
            end
            
            if Thr_n==1
                ni=NumNodes(layer,i);
                
                Total_NN_size=[Total_NN_size,size_counter+ni-2*Q];  %   The total number of random nodes is updating
                
                if i>1
                    Ri=[Ri;2*rand(ni-NumNodes(layer,i-1), Pi)-1];   %   adding new random nodes to the network
                end
                
                Zi_part2=Ri*Yi;
                Zi_part2=normc(Zi_part2);   %   The regularization action to be done at each layer
                Zi=[Zi_part1;Zi_part2];
                Yi_temp=g(Zi);
                
                Oi=LS_ADMM(T_train,Yi_temp,eps_o, mu, kmax);    %   The ADMM solver for constrained least square
                t_hati=Oi*Yi_temp;
                
                train_error=[train_error,Calculate_error(T_train,t_hati)];
                train_accuracy=[train_accuracy,Calculate_accuracy(T_train,t_hati)];
                
                %%%%%%%%%%  Test
                %   Following the same procedure for test data
                Zi_part2_test=Ri*Yi_test;
                Zi_part2_test=normc(Zi_part2_test);
                Zi_test=[Zi_part1_test;Zi_part2_test];
                Yi_test_temp=g(Zi_test);
                t_hati_test=Oi*Yi_test_temp;
                
                test_error=[test_error,Calculate_error(T_test,t_hati_test)];
                test_accuracy=[test_accuracy,Calculate_accuracy(T_test,t_hati_test)];
                
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
                %   plotting accuracy in each step for train and test data (can be plotted only at the end)
                figure(1)
                plot(Total_NN_size,train_accuracy,'b','Linewidth',2);
                hold on; grid on
                plot(Total_NN_size,test_accuracy,'r:','Linewidth',2);
                ylabel('Accuracy','FontName','Times New Roman')
                xlabel('Total number of random nodes','FontName','Times New Roman')
                legend('Training Accuracy','Testing Accuracy','Location','southeast')
                hold off
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
                
                %    checking to see if any of the thresholds has been reached or not
                Thr_n=((train_error(end-1)-train_error(end))/abs(train_error(end-1)))>=eta_n;
                
                if size(NumNodes,2)==1
                    if i==1 && layer>1
                        Thr_l=((train_error(end-1)-train_error(end))/abs(train_error(end-1)))>=eta_l;
                    end
                else
                    if i==1
                        error_temp=train_error(end-1);
                    end
                end
                
            end
            
        end
        
        if size(NumNodes,2)>1
            Thr_l=((error_temp-train_error(end))/abs(error_temp))>=eta_l;
        end
        
        %    updating the variables for the next layer
        Yi=Yi_temp;
        Yi_test=Yi_test_temp;
        Pi=ni;
        NumNode_opt=[NumNode_opt,ni];   %   Optimum number of nodes at this layer
        size_counter=size_counter+ni-2*Q;   % Updating the total number of random nodes at the end of each layer
    end
end

han = gca(figure(1));
set(han,'fontsize',12,'FontName','Times New Roman');
