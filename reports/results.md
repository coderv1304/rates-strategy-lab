SYNTHETIC DATA - pipeline demo only.

| signal         |   IS Sharpe (fold mean) |   OOS Sharpe |   OOS Sharpe @1bp/leg |   OOS ann ret (bp) |   OOS ann vol (bp) |   Sortino |   Max DD (bp) |   Calmar |   Hit rate |   Turnover/yr |   Years |
|:---------------|------------------------:|-------------:|----------------------:|-------------------:|-------------------:|----------:|--------------:|---------:|-----------:|--------------:|--------:|
| macro_residual |                   0.197 |       -0.053 |                -0.272 |             -1.344 |             25.249 |    -0.049 |       -68.159 |   -0.02  |      0.492 |         3.754 |  12.786 |
| pca_slope      |                   0.195 |        0.092 |                -0.07  |              2.644 |             28.6   |     0.095 |       -69.989 |    0.038 |      0.492 |         3.418 |  12.29  |
| momentum       |                   0.011 |       -0.327 |                -0.582 |             -4.208 |             12.851 |    -0.166 |       -71.152 |   -0.059 |      0.458 |         2.058 |  15.552 |

Best OOS signal: **pca_slope**. Bootstrap 95% CI: [-0.29, 0.50]. Circular-shift p = 0.385. Deflated Sharpe probability = 0.168 over 33 variants.
