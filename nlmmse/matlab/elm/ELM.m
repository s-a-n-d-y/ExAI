function [T_hat, Tt_hat, train_error, test_error]=ELM(X, T, Xt, Tt, lam, NumNodes)

% % % % % % % Inputs % % % % % % % %
% X: Train input
% T: Train target
% Xt: Test input
% Tt: Test Target
% lam: regularization hyperparameter
% NumNodes: Number of hidden neurons

% % % % % % % Outputs % % % % % % % 
% T_hat: Train target estimate
% Tt_hat: Test target estimate

%%
a_leaky_RLU=0;      %   set to a small non-zero value if you want to test leaky-RLU
g=@(x) x.*(x >= 0)+a_leaky_RLU*x.*(x < 0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input layer weight matrix: W
[P,N]=size(X);
W=2*rand(NumNodes,P)-1;
b=2*rand(NumNodes,1)-1;
ind=ones(1,N);
B=b(:,ind);
Z=W*X+B;

%% Hidden neorons
Y=g(Z);

% % Output layer weight matrix: O
if NumNodes < N
    O=(T*Y')/(Y*Y'+lam*eye(size(Y,1)));
else
    O=(T/(Y'*Y+lam*eye(size(Y,2))))*Y';
end

T_hat=O*Y;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Testing Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
Nt=size(Xt,2);
ind=ones(1,Nt);
B=b(:,ind);
Zt=W*Xt+B;

Yt=g(Zt);
Tt_hat=O*Yt;

train_error=10*log10(norm(T_hat-T)^2);
test_error=10*log10(norm(Tt_hat-Tt)^2);

return
