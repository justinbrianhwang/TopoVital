# Embedding selection — PENDING researcher decisions

These diagnostics and the PH cost pilot do not select downstream settings.

- Decide how per-segment tau_ami and m_fnn become one Development value (for example, median or mode), including treatment of failures.
- Decide the AMI histogram binning and any other NeuroKit2 estimation settings; this pilot calls `complexity_delay(..., delay_max=100, method="fraser1986")` with library defaults.
- Approve or revise the FNN criteria used in this diagnostic pilot: Rtol=15, Atol=2, <5%, candidate m in [3,5,7].
- Approve or revise the 300-point subsampling rule. The resource pilot uses `np.linspace(0,n-1,300).round()` only for cost measurement.
- Decide the fixed r_max selection rule on Development. The resource pilot uses no maximum edge length to measure upper-bound cost.

The PH pilot uses fallback tau=10 and m=5 only for resource measurement.
