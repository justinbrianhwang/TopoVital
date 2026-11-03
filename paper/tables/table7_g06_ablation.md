# Development only, exploratory; no feature selection

| split | ablation | reference | aggregation | delta_auroc | delta_c0_fpr | delta_c2_fpr |
| --- | --- | --- | --- | --- | --- | --- |
| Development (nested LOSO OOF) | classical_only__minus_A_rhythm | Classical | subject_equal_weight | 0.037 [-0.011, 0.108] | -0.159 | -0.151 |
| Development (nested LOSO OOF) | classical_only__minus_B_qrs_template | Classical | subject_equal_weight | 0.062 [-0.002, 0.152] | -0.240 | -0.204 |
| Development (nested LOSO OOF) | classical_only__minus_C_amplitude | Classical | subject_equal_weight | 0.010 [-0.004, 0.029] | 0.004 | 0.015 |
| Development (nested LOSO OOF) | classical_only__minus_D_dwt | Classical | subject_equal_weight | 0.010 [-0.011, 0.036] | -0.203 | -0.190 |
| Development (nested LOSO OOF) | classical_only__minus_E_psd | Classical | subject_equal_weight | 0.002 [-0.004, 0.012] | -0.012 | -0.009 |
| Development (nested LOSO OOF) | classical_only__minus_F_spectral_entropy | Classical | subject_equal_weight | -0.000 [-0.002, 0.002] | 0.000 | -0.001 |
| Development (nested LOSO OOF) | classical_only__minus_G_sqi | Classical | subject_equal_weight | 0.018 [0.003, 0.039] | -0.044 | -0.055 |
| Development (nested LOSO OOF) | classical_only_no_boundary__minus_A_rhythm | Classical (no boundary) | subject_equal_weight | 0.001 [-0.027, 0.031] | -0.177 | -0.128 |
| Development (nested LOSO OOF) | classical_only_no_boundary__minus_B_qrs_template | Classical (no boundary) | subject_equal_weight | 0.029 [-0.037, 0.095] | -0.126 | -0.087 |
| Development (nested LOSO OOF) | classical_only_no_boundary__minus_C_amplitude | Classical (no boundary) | subject_equal_weight | -0.010 [-0.028, 0.003] | 0.009 | 0.006 |
| Development (nested LOSO OOF) | classical_only_no_boundary__minus_D_dwt | Classical (no boundary) | subject_equal_weight | 0.011 [-0.027, 0.061] | -0.172 | -0.162 |
| Development (nested LOSO OOF) | classical_only_no_boundary__minus_E_psd | Classical (no boundary) | subject_equal_weight | -0.011 [-0.030, 0.005] | 0.020 | 0.014 |
| Development (nested LOSO OOF) | classical_only_no_boundary__minus_F_spectral_entropy | Classical (no boundary) | subject_equal_weight | -0.014 [-0.040, -0.000] | 0.012 | 0.013 |
| Development (nested LOSO OOF) | classical_only_no_boundary__minus_G_sqi | Classical (no boundary) | subject_equal_weight | -0.001 [-0.017, 0.015] | 0.010 | -0.008 |
