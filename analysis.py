# Marimo interactive data analysis notebook
# Author email: 23f3001513@ds.study.iitm.ac.in
#
# How to run:
#   pip install marimo
#   marimo run analysis.py
#
# This notebook demonstrates variable dependencies across cells,
# interactive widgets, and dynamic markdown that reacts to widget state.

import marimo as mo

app = mo.App()

# --- Cell 1: Imports used across the notebook --------------------------------
# Returns mo, numpy and matplotlib for downstream dependency injection.
@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt

# --- Cell 2: UI controls (widgets) -------------------------------------------
# Provides three sliders; their .value properties are consumed by later cells.
@app.cell
def _(mo):
    slope = mo.ui.slider(0.0, 5.0, step=0.1, value=1.5, label="Slope a")
    noise = mo.ui.slider(0.0, 2.0, step=0.1, value=0.4, label="Noise σ")
    n = mo.ui.slider(50, 500, step=10, value=150, label="Sample size n")
    controls = mo.hstack([slope, noise, n])
    controls  # Display the controls in the UI
    return n, noise, slope

# --- Cell 3: Generate data from widgets (depends on Cell 1 & 2) --------------
# Uses numpy from Cell 1 and values from sliders in Cell 2.
@app.cell
def _(np, slope, noise, n):
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, int(n.value))
    y = slope.value * x + rng.normal(0.0, float(noise.value), size=x.shape)
    # Expose x,y for downstream plot and metrics cells.
    return x, y

# --- Cell 4: Fit a simple linear model (depends on Cell 3) -------------------
# Computes least-squares fit; downstream plot and metrics use these coefficients.
@app.cell
def _(np, x, y):
    # Add intercept term for simple OLS using normal equations
    X = np.c_[np.ones_like(x), x]
    beta = np.linalg.pinv(X) @ y
    intercept_hat, slope_hat = beta.tolist()
    return intercept_hat, slope_hat

# --- Cell 5: Dynamic markdown (depends on UI + fit) --------------------------
# Shows a sentence that changes as the user moves the sliders.
@app.cell
def _(mo, slope, noise, slope_hat):
    message = (
        f"**Current simulation:** true slope `a = {slope.value:.2f}`, "
        f"noise `σ = {noise.value:.2f}` → estimated slope "
        f"`â = {slope_hat:.2f}`."
    )
    return mo.md(message)

# --- Cell 6: Visualization (depends on data & fit) ---------------------------
# Plots the simulated points and the fitted regression line.
@app.cell
def _(plt, np, x, y, intercept_hat, slope_hat):
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=16, alpha=0.8, label="Samples")
    y_hat = intercept_hat + slope_hat * x
    ax.plot(x, y_hat, linewidth=2, label="OLS fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Linear relationship with noise")
    ax.legend()
    fig  # Display the figure
    return

# --- Cell 7: Metrics table (depends on data) ---------------------------------
# Computes simple summary metrics to show dependency on n and noise.
@app.cell
def _(np, y, noise, n, mo):
    mse = float(np.mean((y - np.mean(y)) ** 2))
    details = mo.md(
        f"""
**Summary metrics**  
- Observations `n = {int(n.value)}`  
- Sample variance `≈ {np.var(y):.4f}`  
- Baseline MSE (wrt mean): `{mse:.4f}`  
"""
    )
    return details

# --- Cell 8: Complexity note (static; documents analysis intent) -------------
# Demonstrates markdown + math rendered in the app.
@app.cell
def _(mo):
    return mo.md(r"""
### Complexity note
The least-squares estimate via the normal equation uses a matrix pseudo-inverse,
which is typically $O(n)$ to form $X^\\top X$ for our 2-parameter model and
$O(1)$ to invert the resulting $2\\times 2$ matrix.
""")

if __name__ == "__main__":
    app.run()
