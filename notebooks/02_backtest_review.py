# %% [markdown]
# # 02 - Review the walk-forward backtest
# Run `python -m rateslab.pipeline` first so reports/ is populated.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import pandas as pd

from rateslab import config as C

results = pd.read_csv(C.REPORT_DIR / "backtest_results.csv", index_col=0)
results.round(3).T

# %% [markdown]
# ## Fold-by-fold parameters and in-sample vs out-of-sample Sharpe
# If in-sample Sharpe is far above out-of-sample Sharpe in most folds, that is overfitting.

# %%
folds = pd.read_csv(C.REPORT_DIR / "folds_macro_residual.csv", index_col=0, parse_dates=True)
folds[["entry", "exit", "is_sharpe", "oos_sharpe"]].round(2)

# %%
print("Mean IS Sharpe :", round(folds["is_sharpe"].mean(), 2))
print("Mean OOS Sharpe:", round(folds["oos_sharpe"].mean(), 2))
print("Share of folds with positive OOS Sharpe:", round((folds["oos_sharpe"] > 0).mean(), 2))
