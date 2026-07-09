import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =====================================================
# MEMBACA DATA
# =====================================================

df = pd.read_excel("BEBAS DAH.xlsx")

X = df[['Swelling']]
y = df['CBR']

# =====================================================
# MEMBANGUN MODEL SURROGATE
# =====================================================

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

# =====================================================
# METRIK
# =====================================================

r2 = r2_score(y, pred)
mae = mean_absolute_error(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))

a = model.intercept_
b = model.coef_[0]

print("="*50)
print("LINEAR SURROGATE MODEL")
print("="*50)
print(f"CBR = {a:.6f} + ({b:.6f}) × Swelling")
print()
print(f"R²   = {r2:.4f}")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")

# =====================================================
# OUTPUT FOLDER
# =====================================================

os.makedirs("output", exist_ok=True)

# =====================================================
# HASIL KE EXCEL
# =====================================================

hasil = df.copy()
hasil["Prediksi CBR"] = pred
hasil["Residual"] = y - pred

hasil.to_excel("output/hasil_regresi.xlsx", index=False)

# =====================================================
# HEATMAP
# =====================================================

plt.figure(figsize=(6,5))
sns.heatmap(df.corr(),
            annot=True,
            cmap="RdYlBu",
            fmt=".3f",
            linewidths=0.5)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("output/Heatmap.png", dpi=300)
plt.close()

# =====================================================
# SCATTER + REGRESSION
# =====================================================

plt.figure(figsize=(7,5))

plt.scatter(X, y,
            s=80,
            color='royalblue',
            label='Data')

urut = np.argsort(X.values.flatten())

plt.plot(X.values.flatten()[urut],
         pred[urut],
         color='red',
         linewidth=2.5,
         label='Linear Regression')

plt.xlabel("Swelling (%)")
plt.ylabel("CBR (%)")
plt.title("Swelling vs CBR")
plt.legend()

plt.grid(True)

plt.tight_layout()
plt.savefig("output/Scatter_Regresi.png", dpi=300)
plt.close()

# =====================================================
# RESIDUAL
# =====================================================

plt.figure(figsize=(7,5))

plt.scatter(pred, y-pred, s=80)

plt.axhline(0,
            color='red',
            linestyle='--')

plt.xlabel("Predicted")
plt.ylabel("Residual")

plt.title("Residual Plot")

plt.grid(True)

plt.tight_layout()

plt.savefig("output/Residual.png", dpi=300)

plt.close()

# =====================================================
# PREDIKSI VS AKTUAL
# =====================================================

plt.figure(figsize=(6,6))

plt.scatter(y, pred,
            s=80)

minv = min(y.min(), pred.min())
maxv = max(y.max(), pred.max())

plt.plot([minv,maxv],
         [minv,maxv],
         'r--')

plt.xlabel("Actual CBR")
plt.ylabel("Predicted CBR")

plt.title("Actual vs Predicted")

plt.grid(True)

plt.tight_layout()

plt.savefig("output/Prediksi_vs_Aktual.png", dpi=300)

plt.close()

# =====================================================
# RINGKASAN
# =====================================================

with open("output/ringkasan.txt","w") as f:

    f.write("LINEAR SURROGATE MODEL\n")
    f.write("=======================\n\n")
    f.write(f"CBR = {a:.6f} + ({b:.6f})*Swelling\n\n")
    f.write(f"R2   = {r2:.4f}\n")
    f.write(f"MAE  = {mae:.4f}\n")
    f.write(f"RMSE = {rmse:.4f}\n")

print("\nSeluruh hasil berhasil disimpan pada folder OUTPUT")