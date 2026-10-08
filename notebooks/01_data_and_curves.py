# %% [markdown]
# # 01 - Data, curve shape and NSS fit
# Open in VS Code and click "Run Cell" above each `# %%` block (Jupyter extension required).

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rateslab import config as C
from rateslab.curves import fit_nss_day, nss_yield
from rateslab.data_loader import slope_bp

yields = pd.read_csv(C.DATA_PROC / "yields.csv", index_col=0, parse_dates=True)
slope = slope_bp(yields)
yields.tail()

# %% [markdown]
# ## Fluency check: do this without looking anything up
# 1. Resample 2s10s to month-end and plot it. 2. Show the 5 largest daily 10Y moves.
# 3. Compute the correlation matrix of daily changes.

# %%
slope.resample("ME").last().plot(figsize=(10, 3.5), title="2s10s, month-end (bp)")
plt.show()
yields["10Y"].diff().abs().nlargest(5)

# %%
(yields.diff().dropna() * 100).corr().round(2)

# %% [markdown]
# ## Nelson-Siegel-Svensson fit on one date

# %%
date = yields.index[-1]
taus = np.array([C.TENOR_YEARS[c] for c in yields.columns])
params, rmse = fit_nss_day(taus, yields.loc[date].values)
print(date.date(), "RMSE (bp):", round(rmse, 2))
grid = np.linspace(0.1, 30, 200)
plt.scatter(taus, yields.loc[date], color="k", label="par yields")
plt.plot(grid, nss_yield(grid, *params), label="NSS fit")
plt.legend()
plt.show()
