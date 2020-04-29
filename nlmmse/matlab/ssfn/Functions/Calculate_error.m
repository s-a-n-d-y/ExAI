function error=Calculate_error(T,T_hat)
%   Calculate error
%error=20*log10(norm(T-T_hat,'fro')/norm(T,'fro'));

%error=10*log10(norm(T-T_hat)^2);

% error=(norm(T-T_hat)^2);

diff = T-T_hat;
diff_norm = zeros(size(diff,2),1);
parfor i=1:size(diff,2)
    diff_norm(i) = norm(diff(:,i))^2;
end
error=mean(diff_norm);

end