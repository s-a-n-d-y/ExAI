tic;
regression = ["ra" "rb_a_1" "rb_a_10" "rc" "rd"];
parfor i=1:length(regression)
    main_R(regression(i));
end

main_C_N("ca");

main_C_DN("cda");
toc;