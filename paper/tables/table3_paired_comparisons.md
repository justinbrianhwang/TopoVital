# Paired model comparisons; primary row marked

| pair | primary | rule | metric | Development (nested LOSO OOF), subject_equal_weight, difference | Development (nested LOSO OOF), window_level, difference | Independent Test, subject_equal_weight, difference | Independent Test, window_level, difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Classical+PH − Classical | True | primary | stratified_auroc | -0.026 [-0.091, 0.021] | -0.028 [-0.069, 0.007] | -0.122 [-0.302, -0.017] | -0.037 [-0.085, -0.004] |
| Classical+PH − PH only | False | primary | stratified_auroc | 0.377 [0.225, 0.564] | 0.269 [0.210, 0.327] | 0.234 [0.145, 0.321] | 0.313 [0.256, 0.361] |
| Classical − PH only | False | primary | stratified_auroc | 0.404 [0.254, 0.557] | 0.297 [0.255, 0.337] | 0.357 [0.219, 0.511] | 0.349 [0.293, 0.422] |
| Classical − Classical (no boundary) | False | primary | stratified_auroc | 0.039 [-0.003, 0.085] | 0.098 [0.018, 0.153] | 0.023 [0.002, 0.049] | 0.028 [0.010, 0.046] |
| Classical+PH − Classical+PH (no boundary) | False | primary | stratified_auroc | -0.005 [-0.096, 0.061] | 0.075 [0.005, 0.124] | 0.018 [-0.018, 0.051] | 0.028 [-0.019, 0.070] |
| Classical+PH (no boundary) − Classical (no boundary) | False | primary | stratified_auroc | 0.018 [-0.000, 0.040] | -0.005 [-0.046, 0.020] | -0.118 [-0.295, -0.004] | -0.036 [-0.117, 0.012] |
