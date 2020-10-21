function O=LS_ADMM(T, Y, eps_o, mu, kmax)
Lam=zeros(size(T,1),size(Y,1));
YYT=Y*Y';
temp=inv((YYT+(1/mu)*(eye(size(Y,1)))));
TYT=T*Y';
Z=Lam;

Lam_norm=zeros(1,kmax);
for iter=1:kmax
    % O-update
    O=(TYT+(1/mu)*(Z+Lam))*temp;
    % Z-update
    Z=O-Lam;
    nz=norm(Z,'fro');
    if nz > eps_o
        Z=Z*(eps_o/nz);
    end
    % Lam-update
    Lam=Lam+Z-O;
%     
%     Lam_norm(iter)=norm(Lam,'fro');
%     figure(3)
%     plot(1:iter,Lam_norm(1:iter))
%     grid on
%     xlabel('Iteration number')
%     ylabel('Error')
%     drawnow
    
end