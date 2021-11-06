approximate gp
➜  offboard_code git:(master) ✗ python3 test_mpc_gpy.py
dimension of x_1d before prune: (25600, 1)
dimension of x_2d before prune: (25600, 1)
dimension of x before prune: (25600, 2)
dimension of x: (25600, 2)
dimension of y: (25600,)
dimension of x_1d before prune: (25600, 1)
dimension of x_2d before prune: (25600, 1)
dimension of x before prune: (25600, 2)
dimension of x: (25600, 2)
dimension of y: (25600,)
dimension of x_1d before prune: (25600, 1)
dimension of x_2d before prune: (25600, 1)
dimension of x before prune: (25600, 2)
dimension of x: (25600, 2)
dimension of y: (25600,)
first predict time is:  0.023804664611816406
second predict time is:  0.007836103439331055
third predict time is:  0.007576942443847656

exact gp
➜  offboard_code git:(master) ✗ python3 test_mpc_gpy.py
train_x_max_1: 0.5189882887329315
train_x_min_1: -0.495962671934747
train_x_max_2: 0.4739474356174469
train_x_min_2: 0.2524917721748352
dimension of x_1d before prune: (25600, 1)
dimension of x_2d before prune: (25600, 1)
dimension of x before prune: (25600, 2)
dimension of y before prune: (25600,)
test_x.shape:(15360, 2)
x_train shape torch.Size([10240, 2])
y_train shape torch.Size([10240])
train_x_max_1: 0.3822494562516663
train_x_min_1: -0.47540126389161846
train_x_max_2: 0.4739474356174469
train_x_min_2: 0.2524917721748352
dimension of x_1d before prune: (25600, 1)
dimension of x_2d before prune: (25600, 1)
dimension of x before prune: (25600, 2)
dimension of y before prune: (25600,)
test_x.shape:(15360, 2)
x_train shape torch.Size([10240, 2])
y_train shape torch.Size([10240])
train_x_max_1: 0.19114933257223385
train_x_min_1: -0.2095782532755353
train_x_max_2: 0.4739474356174469
train_x_min_2: 0.2524917721748352
dimension of x_1d before prune: (25600, 1)
dimension of x_2d before prune: (25600, 1)
dimension of x before prune: (25600, 2)
dimension of y before prune: (25600,)
test_x.shape:(15360, 2)
x_train shape torch.Size([10240, 2])
y_train shape torch.Size([10240])
first predict time is:  1.8051745891571045
second predict time is:  0.003847360610961914
third predict time is:  0.0035970211029052734

