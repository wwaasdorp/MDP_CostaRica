% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 661.669894655137000 ; 662.828455220482624 ];

%-- Principal point:
cc = [ 306.096010906942354 ; 240.789943718702290 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ -0.264245252903117 ; 0.226442226819631 ; 0.000202199025333 ; 0.000228356495298 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 1.179062101992759 ; 1.265600866332926 ];

%-- Principal point uncertainty:
cc_error = [ 2.384293528759300 ; 2.174690927938554 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.009334527090312 ; 0.038260496420282 ; 0.000517895581176 ; 0.000531660126468 ; 0.000000000000000 ];

%-- Image size:
nx = 640;
ny = 480;


%-- Various other variables (may be ignored if you do not use the Matlab Calibration Toolbox):
%-- Those variables are used to control which intrinsic parameters should be optimized

n_ima = 20;						% Number of calibration images
est_fc = [ 1 ; 1 ];					% Estimation indicator of the two focal variables
est_aspect_ratio = 1;				% Estimation indicator of the aspect ratio fc(2)/fc(1)
center_optim = 1;					% Estimation indicator of the principal point
est_alpha = 0;						% Estimation indicator of the skew coefficient
est_dist = [ 1 ; 1 ; 1 ; 1 ; 0 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ 1.652732e+00 ; 1.646584e+00 ; -6.684721e-01 ];
Tc_1  = [ -1.816233e+02 ; -8.127242e+01 ; 8.574056e+02 ];
omc_error_1 = [ 2.743946e-03 ; 3.571467e-03 ; 4.547457e-03 ];
Tc_error_1  = [ 3.096543e+00 ; 2.841622e+00 ; 1.579182e+00 ];

%-- Image #2:
omc_2 = [ 1.845891e+00 ; 1.895258e+00 ; -3.966240e-01 ];
Tc_2  = [ -1.585894e+02 ; -1.571082e+02 ; 7.622968e+02 ];
omc_error_2 = [ 2.897253e-03 ; 3.563277e-03 ; 5.515698e-03 ];
Tc_error_2  = [ 2.767812e+00 ; 2.523575e+00 ; 1.549522e+00 ];

%-- Image #3:
omc_3 = [ 1.739650e+00 ; 2.071858e+00 ; -5.051114e-01 ];
Tc_3  = [ -1.289290e+02 ; -1.723141e+02 ; 7.803303e+02 ];
omc_error_3 = [ 2.650118e-03 ; 3.774328e-03 ; 5.700450e-03 ];
Tc_error_3  = [ 2.829819e+00 ; 2.582519e+00 ; 1.487997e+00 ];

%-- Image #4:
omc_4 = [ 1.826653e+00 ; 2.110577e+00 ; -1.099471e+00 ];
Tc_4  = [ -6.820014e+01 ; -1.525924e+02 ; 7.836082e+02 ];
omc_error_4 = [ 2.375044e-03 ; 3.895173e-03 ; 5.326924e-03 ];
Tc_error_4  = [ 2.847412e+00 ; 2.576495e+00 ; 1.199763e+00 ];

%-- Image #5:
omc_5 = [ 1.077082e+00 ; 1.917430e+00 ; -2.540544e-01 ];
Tc_5  = [ -9.588660e+01 ; -2.269486e+02 ; 7.415613e+02 ];
omc_error_5 = [ 2.321175e-03 ; 3.641237e-03 ; 4.078149e-03 ];
Tc_error_5  = [ 2.716574e+00 ; 2.459798e+00 ; 1.463970e+00 ];

%-- Image #6:
omc_6 = [ -1.701512e+00 ; -1.932874e+00 ; -7.953316e-01 ];
Tc_6  = [ -1.510290e+02 ; -7.829463e+01 ; 4.481140e+02 ];
omc_error_6 = [ 2.246668e-03 ; 3.638883e-03 ; 4.920616e-03 ];
Tc_error_6  = [ 1.628046e+00 ; 1.520457e+00 ; 1.257233e+00 ];

%-- Image #7:
omc_7 = [ 1.991142e+00 ; 1.931116e+00 ; 1.311379e+00 ];
Tc_7  = [ -8.512075e+01 ; -7.651121e+01 ; 4.445033e+02 ];
omc_error_7 = [ 4.299380e-03 ; 2.196408e-03 ; 5.137210e-03 ];
Tc_error_7  = [ 1.640159e+00 ; 1.489999e+00 ; 1.325036e+00 ];

%-- Image #8:
omc_8 = [ 1.957075e+00 ; 1.824172e+00 ; 1.325608e+00 ];
Tc_8  = [ -1.727514e+02 ; -1.023108e+02 ; 4.668263e+02 ];
omc_error_8 = [ 4.084228e-03 ; 2.231468e-03 ; 4.909835e-03 ];
Tc_error_8  = [ 1.797913e+00 ; 1.620765e+00 ; 1.492918e+00 ];

%-- Image #9:
omc_9 = [ -1.367856e+00 ; -1.985915e+00 ; 3.197103e-01 ];
Tc_9  = [ -5.269496e+00 ; -2.231768e+02 ; 7.344192e+02 ];
omc_error_9 = [ 2.789176e-03 ; 3.582876e-03 ; 4.625670e-03 ];
Tc_error_9  = [ 2.681506e+00 ; 2.423919e+00 ; 1.522240e+00 ];

%-- Image #10:
omc_10 = [ -1.517744e+00 ; -2.091692e+00 ; 1.862023e-01 ];
Tc_10  = [ -3.375879e+01 ; -2.979547e+02 ; 8.665898e+02 ];
omc_error_10 = [ 3.393792e-03 ; 4.062541e-03 ; 6.153033e-03 ];
Tc_error_10  = [ 3.221056e+00 ; 2.879916e+00 ; 2.020712e+00 ];

%-- Image #11:
omc_11 = [ -1.794197e+00 ; -2.067964e+00 ; -4.838637e-01 ];
Tc_11  = [ -1.544659e+02 ; -2.331239e+02 ; 7.093752e+02 ];
omc_error_11 = [ 3.038325e-03 ; 3.833536e-03 ; 6.591660e-03 ];
Tc_error_11  = [ 2.638876e+00 ; 2.464133e+00 ; 1.995774e+00 ];

%-- Image #12:
omc_12 = [ -1.839776e+00 ; -2.090491e+00 ; -5.199217e-01 ];
Tc_12  = [ -1.363775e+02 ; -1.753465e+02 ; 6.089474e+02 ];
omc_error_12 = [ 2.601921e-03 ; 3.701771e-03 ; 6.104675e-03 ];
Tc_error_12  = [ 2.247133e+00 ; 2.082814e+00 ; 1.669715e+00 ];

%-- Image #13:
omc_13 = [ -1.919192e+00 ; -2.119523e+00 ; -5.988288e-01 ];
Tc_13  = [ -1.352905e+02 ; -1.418790e+02 ; 5.484861e+02 ];
omc_error_13 = [ 2.432797e-03 ; 3.669768e-03 ; 6.006224e-03 ];
Tc_error_13  = [ 2.017974e+00 ; 1.864975e+00 ; 1.516781e+00 ];

%-- Image #14:
omc_14 = [ -1.954748e+00 ; -2.127785e+00 ; -5.895144e-01 ];
Tc_14  = [ -1.259162e+02 ; -1.356283e+02 ; 4.943712e+02 ];
omc_error_14 = [ 2.286373e-03 ; 3.589227e-03 ; 5.874198e-03 ];
Tc_error_14  = [ 1.821974e+00 ; 1.680175e+00 ; 1.361806e+00 ];

%-- Image #15:
omc_15 = [ -2.108920e+00 ; -2.253739e+00 ; -4.957979e-01 ];
Tc_15  = [ -2.017190e+02 ; -1.330666e+02 ; 4.792403e+02 ];
omc_error_15 = [ 2.622057e-03 ; 3.322192e-03 ; 6.347036e-03 ];
Tc_error_15  = [ 1.796244e+00 ; 1.669700e+00 ; 1.457504e+00 ];

%-- Image #16:
omc_16 = [ 1.884415e+00 ; 2.331612e+00 ; -1.589216e-01 ];
Tc_16  = [ -1.932201e+01 ; -1.685750e+02 ; 7.002722e+02 ];
omc_error_16 = [ 3.592290e-03 ; 3.793990e-03 ; 7.752635e-03 ];
Tc_error_16  = [ 2.540560e+00 ; 2.297737e+00 ; 1.738348e+00 ];

%-- Image #17:
omc_17 = [ -1.614493e+00 ; -1.957668e+00 ; -3.520423e-01 ];
Tc_17  = [ -1.375535e+02 ; -1.374591e+02 ; 4.933804e+02 ];
omc_error_17 = [ 2.262290e-03 ; 3.465944e-03 ; 4.874178e-03 ];
Tc_error_17  = [ 1.798390e+00 ; 1.666249e+00 ; 1.212190e+00 ];

%-- Image #18:
omc_18 = [ -1.345751e+00 ; -1.706821e+00 ; -3.061455e-01 ];
Tc_18  = [ -1.860989e+02 ; -1.557986e+02 ; 4.485036e+02 ];
omc_error_18 = [ 2.574777e-03 ; 3.382118e-03 ; 3.868610e-03 ];
Tc_error_18  = [ 1.649297e+00 ; 1.532944e+00 ; 1.176323e+00 ];

%-- Image #19:
omc_19 = [ -1.925287e+00 ; -1.841831e+00 ; -1.441336e+00 ];
Tc_19  = [ -1.087415e+02 ; -7.861849e+01 ; 3.386713e+02 ];
omc_error_19 = [ 2.232461e-03 ; 3.920133e-03 ; 4.975414e-03 ];
Tc_error_19  = [ 1.277562e+00 ; 1.168105e+00 ; 1.106090e+00 ];

%-- Image #20:
omc_20 = [ 1.891448e+00 ; 1.593230e+00 ; 1.471079e+00 ];
Tc_20  = [ -1.462121e+02 ; -8.702790e+01 ; 4.007252e+02 ];
omc_error_20 = [ 4.142701e-03 ; 2.284271e-03 ; 4.447635e-03 ];
Tc_error_20  = [ 1.560086e+00 ; 1.388737e+00 ; 1.334161e+00 ];

