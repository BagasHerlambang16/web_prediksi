import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def prediksi_arima(tahun_awal, bulan_awal, tahun_akhir, bulan_selesai):
    # Load dan siapkan data
    df = pd.read_excel('data/datapendudukbulanan.xlsx')
    bulan_map = {
        "Januari": 1, "Februari": 2, "Maret": 3, "April": 4, "Mei": 5, "Juni": 6,
        "Juli": 7, "Agustus": 8, "September": 9, "Oktober": 10, "November": 11, "Desember": 12
    }
    df['Bulan_Angka'] = df['Bulan'].map(bulan_map)
    df['Tanggal'] = pd.to_datetime(dict(year=df['Tahun'], month=df['Bulan_Angka'], day=1))
    df = df[['Tanggal', 'Jumlah']].set_index('Tanggal')
    df = df.groupby(df.index).mean()

    # Smoothing ringan
    df['Jumlah'] = df['Jumlah'].rolling(window=3, min_periods=1).mean()

    # Model ARIMA
    model = ARIMA(df['Jumlah'], order=(1, 1, 1))
    model_fit = model.fit()

    # Hitung jumlah bulan yang akan diprediksi
    start_date = pd.Timestamp(year=tahun_awal, month=bulan_awal, day=1)
    end_date = pd.Timestamp(year=tahun_akhir, month=bulan_selesai, day=1)
    if end_date <= start_date:
        raise ValueError("Bulan dan Tahun Selesai harus lebih Besar Daripada Bulan dan Tahun Mulai.")
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

    # Prediksi awal
    forecast = model_fit.forecast(steps=total_months)

    # Koreksi agar prediksi tidak turun
    adjusted_forecast = []
    last_value = df['Jumlah'].iloc[-1]

    for val in forecast:
        val = float(val)
        if val < last_value:
            val = last_value + np.random.uniform(5, 20)  # tambahkan kenaikan acak
        adjusted_forecast.append(val)
        last_value = val  # update nilai terakhir untuk pembanding

    # Format hasil
    tanggal_prediksi = pd.date_range(start=start_date, periods=total_months, freq='MS')
    hasil = {tgl.strftime('%B %Y'): int(round(jml)) for tgl, jml in zip(tanggal_prediksi, adjusted_forecast)}

    return hasil
