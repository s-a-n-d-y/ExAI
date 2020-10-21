%%  All the materials available in this document is to reproduce the results published in the following paper:
% 
%   S. Chatterjee, A. M. Javid, M. Sadeghi, S. Kikuta, P. P. Mitra, M. Skoglund, 
%   "SSFN: Low Complexity Self Size-estimating Feed-forward Neural Network using Layer-wise Convex Optimization", 2019
% 
%%  The codes is organized as follows:
% 
%   SSFN_Performance.m is used to acheive the performance results of SSFN 
%   shown in Table 2 and Table 4 in Section 3
% 
%               SSFN_Performance.m        ->      Table 2 and 4    
% 
%   SSFN_Architecture.m is used to acheive the performance results of SSFN
%   shown in Figure 2 and Table 3 in Section 3
% 
%               SSFN_Architecture.m       ->      Figure 2 and Table 3
% 
%   SSFN_Behavior.m is used to acheive the performance results of SSFN
%   shown in Figure 3 in Section 3
% 
%               SSFN_Behavior.m       	 ->      Figure 3    
%   
%   The files available in the "Functions" folder are as follows:
% 
%   Calculate_accuracy  :   Function for calculating accuracy for a given estimation of target 
%                              
%   Calculate_error     :   Function for calculating NME for a given estimation of target
%                           
%   Load_dataset        :   Function for loading the dataset listed in Table I
%                           
%   LS_ADMM             :   Function for solving constrained least-square problem 
% 
%   LS:                 :   Function for simulating Regularized Least-square
%
%	shadedErrorBar		:	Function for generating Figure 2
%
%   SSFN                :   Function for simulating Self Size-estimating Feed-forward Network
% 
%%  Notes:
%   
%   In "Datasets" folder, you find the used datasets in our experiments. This
%   folder must be placed in the same directory as the codes.
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Contact:    Saikat Chatterjee (sach@kth.se), Alireza Javid (almj@kth.se) 
%
% 	April 2019
