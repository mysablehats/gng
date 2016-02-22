%start script
%load('../share/local_uniform_2d.mat')
%pkg load statistics
%load('allllllllll.mat')
tic
[A, C] = gng_lax(data_val);
toc