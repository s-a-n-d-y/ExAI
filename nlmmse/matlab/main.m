close all;clear;clc;
tic;

% Experiment for data type - A
% regression_exp_A = ["ra_a" "rb_a_1_a" "rb_a_10_a" "rc_a" "rd_a"];
% parfor i=1:length(regression_exp_A)
%     main_R(regression_exp_A(i));
% end
% main_C_N("ca_a");
% main_C_DN("cda_a");

% Experiment for data type - B
% regression_exp_B = ["ra_b" "rb_b" "rc_b" "rd_b"];
% parfor i=1:length(regression_exp_B)
%     main_R(regression_exp_B(i));
% end
main_C_N("ca_b");
main_C_DN("cda_b");

toc;