import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Global style
plt.rcParams.update({
    "font.size":       12,
    "axes.titlesize":  12,
    "axes.labelsize":  12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.family":     "serif",
    "figure.figsize":  (6.0, 3.6),
    "savefig.dpi":     800,
})
sns.set_style('whitegrid')

# Read CSV files (replace with your filenames)
df_alpha = pd.read_csv('robust_analysis.csv')
df_mass  = pd.read_csv('sensitivity_analysis.csv')

# ─── Alpha sensitivity analysis ────────────────────────────────────────────
plt.figure()
plt.plot(df_alpha['alpha'], df_alpha['RMSE_q1'], label='Joint 1 RMSE', linewidth=1.8)
plt.plot(df_alpha['alpha'], df_alpha['RMSE_q2'], label='Joint 2 RMSE', linewidth=1.8)
plt.xlabel('Alpha (α)')
plt.ylabel('RMSE (rad)')
plt.title('Robustness Analysis: RMSE vs Alpha')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/alpha_sensitivity.png', dpi=800)
plt.show()

# ─── Mass error sensitivity analysis ───────────────────────────────────────
plt.figure()
plt.plot(df_mass['mass_error_percent'], df_mass['RMSE_q1'], label='Joint 1 RMSE', linewidth=1.8)
plt.plot(df_mass['mass_error_percent'], df_mass['RMSE_q2'], label='Joint 2 RMSE', linewidth=1.8)
plt.xlabel('Mass Error (%)')
plt.ylabel('RMSE (rad)')
plt.title('Parameter Sensitivity Analysis: RMSE vs Mass Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/mass_error_sensitivity.png', dpi=800)
plt.show()
