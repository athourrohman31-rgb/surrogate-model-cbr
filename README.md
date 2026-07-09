# Linear Surrogate Model for CBR Prediction

## Deskripsi

Repository ini berisi implementasi model surrogate linear menggunakan Python untuk memprediksi nilai California Bearing Ratio (CBR) berdasarkan nilai Swelling pada tanah lempung ekspansif.

## Tujuan

- Membangun model regresi linear antara Swelling dan CBR.
- Menghitung performa model menggunakan R², MAE, dan RMSE.
- Membuat visualisasi berupa heatmap korelasi, scatter plot, residual plot, dan grafik Actual vs Predicted.

## Struktur Folder

```
.
├── data/
│   └── BEBAS DAH.xlsx
├── output/
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── LICENSE-DATA.md
```

## Cara Menjalankan

Install library

```bash
pip install -r requirements.txt
```

Jalankan program

```bash
python main.py
```

## Output

Program akan menghasilkan:

- Heatmap korelasi
- Scatter plot regresi
- Residual plot
- Actual vs Predicted plot
- File Excel hasil prediksi
- Ringkasan model