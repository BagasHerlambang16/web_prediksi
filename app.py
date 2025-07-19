from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, make_response, send_file
)
import mysql.connector
import pandas as pd
import numpy as np
import json
import datetime
from io import BytesIO
from datetime import date, datetime
import openpyxl

# Model & Metrics
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from arima_model import prediksi_arima

app = Flask(__name__)
app.secret_key = 'rahasia123'

# ===================== 🔧 KONEKSI DATABASE =====================
def koneksi_mysql():
    return mysql.connector.connect(
        host="sql12.freesqldatabase.com",
        user="sql12790786",
        password="FcXW2avwGb",
        database="sql12790786"
    )


# ===================== 🔐 LOGIN HANDLER =====================
def cek_login(email, password):
    conn = koneksi_mysql()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM user WHERE email = %s AND password = %s", (email, password))
    user = cursor.fetchone()
    conn.close()
    return user


# ===================== 📊 EVALUASI SEMUA MODEL =====================
def evaluasi_semua_model():
    df = pd.read_excel('data/datapendudukbulanan.xlsx')
    bulan_mapping = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
        'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }

    df['Bulan'] = df['Bulan'].map(bulan_mapping)
    df.dropna(subset=['Bulan'], inplace=True)
    df['Tanggal'] = pd.to_datetime(dict(year=df['Tahun'], month=df['Bulan'], day=1))
    df.set_index('Tanggal', inplace=True)
    df.sort_index(inplace=True)

    data = df['Jumlah']
    split_idx = int(len(data) * 0.8)
    train, test = data[:split_idx], data[split_idx:]

    def evaluate_model(true, predicted):
        true, predicted = np.array(true), np.array(predicted)
        mask = (~np.isnan(true)) & (~np.isnan(predicted)) & (~np.isinf(true)) & (~np.isinf(predicted))
        true, predicted = true[mask], predicted[mask]
        if len(true) == 0:
            return {'RMSE': np.nan, 'R2': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'Accuracy': np.nan}

        safe_true = np.where(true == 0, 1e-10, true)
        return {
            'RMSE': round(np.sqrt(mean_squared_error(true, predicted)), 2),
            'R2': round(r2_score(true, predicted), 2),
            'MAE': round(mean_absolute_error(true, predicted), 2),
            'MAPE': round(np.mean(np.abs((true - predicted) / safe_true)) * 100, 2),
            'Accuracy': round((1 - mean_absolute_error(true, predicted) / np.mean(safe_true)) * 100, 2)
        }

    # Model ARIMA
    arima_pred = ARIMA(train, order=(1, 1, 1)).fit().forecast(len(test))
    sarima_pred = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False).forecast(len(test))
    es_pred = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit().forecast(len(test))

    min_len = min(len(test), len(arima_pred), len(sarima_pred), len(es_pred))
    hybrid_pred = (arima_pred[:min_len].values + sarima_pred[:min_len].values + es_pred[:min_len].values) / 3

    return {
        'ARIMA': evaluate_model(test, arima_pred),
        'SARIMA': evaluate_model(test, sarima_pred),
        'Exponential Smoothing': evaluate_model(test, es_pred),
        'HYBRID': evaluate_model(test.values[:min_len], hybrid_pred)
    }


# ===================== 🌐 ROUTES =====================
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Validasi panjang password minimal 8 karakter
        if len(password) < 8:
            return render_template('login.html', error="Kata sandi minimal 8 karakter")

        user = cek_login(email, password)
        if user:
            session['user'] = user['username']
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Email atau Password Salah")

    return render_template('login.html')



@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')


@app.route('/pertumbuhan')
def pertumbuhan():
    df = pd.read_excel('data/datapendudukbulanan.xlsx')
    df['Bulan'] = df['Bulan'].map({
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
        'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    })
    df.dropna(subset=['Bulan'], inplace=True)
    df['Tanggal'] = pd.to_datetime(dict(year=df['Tahun'], month=df['Bulan'], day=1))
    df.set_index('Tanggal', inplace=True)
    df.sort_index(inplace=True)
    histori_pertahun = df.groupby(df.index.year)['Jumlah'].mean().astype(int).to_dict()
    return render_template('pertumbuhan.html', histori=histori_pertahun)


@app.route('/perbandingan')
def perbandingan():
    hasil = evaluasi_semua_model()

    # Baca data
    df = pd.read_excel('data/datapendudukbulanan.xlsx')
    df['Bulan'] = df['Bulan'].map({
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
        'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    })
    df.dropna(subset=['Bulan'], inplace=True)
    df['Tanggal'] = pd.to_datetime(dict(year=df['Tahun'], month=df['Bulan'], day=1))
    df.set_index('Tanggal', inplace=True)
    df.sort_index(inplace=True)

    # Ambil kolom jumlah penduduk
    data = df['Jumlah']
    split_idx = int(len(data) * 0.8)
    train, test = data[:split_idx], data[split_idx:]

    # Prediksi dengan ARIMA, SARIMA, dan Exponential Smoothing
    arima_pred = ARIMA(train, order=(1, 1, 1)).fit().forecast(len(test))
    sarima_pred = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False).forecast(len(test))
    es_pred = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit().forecast(len(test))

    # Gabungkan hasil prediksi
    min_len = min(len(test), len(arima_pred), len(sarima_pred), len(es_pred))
    hybrid_pred = (arima_pred[:min_len].values + sarima_pred[:min_len].values + es_pred[:min_len].values) / 3

    # Map bulan ke nama
    bulan_map_reverse = {
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 5: 'Mei', 6: 'Juni',
        7: 'Juli', 8: 'Agustus', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
    }
    labels = [f"{bulan_map_reverse[dt.month]} {dt.year}" for dt in test.index[:min_len]]

    # Membulatkan hasil prediksi dan data aktual
    chart_data = {
        'labels': labels,
        'data': {
            'Actual': [int(round(val)) for val in test.values[:min_len]],
            'ARIMA': [int(round(val)) for val in arima_pred[:min_len]],
            'SARIMA': [int(round(val)) for val in sarima_pred[:min_len]],
            'Exponential Smoothing': [int(round(val)) for val in es_pred[:min_len]],
            'HYBRID': [int(round(val)) for val in hybrid_pred]
        }
    }

    return render_template('perbandingan.html', hasil=hasil, chart_data=json.dumps(chart_data))

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    if request.method == 'POST':
        try:
            bulan_mulai = int(request.form['bulan_mulai'])
            tahun_mulai = int(request.form['tahun_mulai'])
            bulan_selesai = int(request.form['bulan_selesai'])
            tahun_selesai = int(request.form['tahun_selesai'])
            
            
            # Validasi tahun_selesai
            if tahun_selesai > 2225:
                flash("Sistem hanya dapat melakukan prediksi hingga maksimal tahun 2225.", "error")
                return redirect(url_for('prediksi'))


            hasil = prediksi_arima(tahun_mulai, bulan_mulai, tahun_selesai, bulan_selesai)

            # Data untuk grafik
            chart_data = {
                "labels": list(hasil.keys()),
                "data": list(hasil.values())
            }

            return render_template('prediksi.html', hasil=hasil, chart_data=chart_data)

        except ValueError as ve:
            flash(str(ve), 'error')
            return redirect(url_for('prediksi'))

    return render_template('prediksi.html')


@app.route('/simpan_prediksi', methods=['POST'])
def simpan_prediksi():
    data = request.form.get('data_prediksi')
    if not data or 'user' not in session:
        flash("Data tidak valid atau belum login.", "error")
        return redirect(url_for('prediksi'))

    try:
        conn = koneksi_mysql()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id FROM user WHERE username = %s", (session['user'],))
        user = cursor.fetchone()
        if not user:
            flash("Pengguna tidak ditemukan.", "error")
            return redirect(url_for('prediksi'))

        cursor.execute("""
            INSERT INTO prediksi (id_user, tanggal_prediksi, jumlah)
            VALUES (%s, %s, %s)
        """, (user['id'], date.today(), data))

        conn.commit()
        conn.close()
        flash("Data Prediksi Berhasil Disimpan.", "success")

    except Exception as e:
        flash(f"Gagal menyimpan data: {e}", "error")

    return redirect(url_for('prediksi'))


@app.route('/riwayat')
def riwayat():
    if 'user' not in session:
        flash("Silakan login terlebih dahulu", "error")
        return redirect(url_for('login'))

    try:
        conn = koneksi_mysql()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT p.id, u.username, p.tanggal_prediksi, p.jumlah
            FROM prediksi p
            JOIN user u ON p.id_user = u.id
            ORDER BY p.created_at ASC
        """)
        rows = cursor.fetchall()

        bulan_map = {
            'January': 'Januari', 'February': 'Februari', 'March': 'Maret',
            'April': 'April', 'May': 'Mei', 'June': 'Juni',
            'July': 'Juli', 'August': 'Agustus', 'September': 'September',
            'October': 'Oktober', 'November': 'November', 'December': 'Desember'
        }

        def parse_bulan_tahun(bt):
            bulan, tahun = bt.split()
            return datetime(int(tahun), list(bulan_map.keys()).index(bulan) + 1, 1)

        for row in rows:
            try:
                hasil = json.loads(row['jumlah'])
                keys = list(hasil.keys())
                if keys:
                    awal, akhir = sorted(keys, key=parse_bulan_tahun)[0], sorted(keys, key=parse_bulan_tahun)[-1]
                    row['rentang'] = f"{bulan_map[awal.split()[0]]} {awal.split()[1]} - {bulan_map[akhir.split()[0]]} {akhir.split()[1]}"
                else:
                    row['rentang'] = "-"
            except:
                row['rentang'] = "-"

        conn.close()
        return render_template('riwayat.html', prediksi_list=rows)

    except Exception as e:
        flash("Gagal memuat data riwayat", "error")
        return redirect(url_for('dashboard'))


@app.route('/riwayat/<int:prediksi_id>')
def lihat_prediksi(prediksi_id):
    conn = koneksi_mysql()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT p.*, u.username FROM prediksi p JOIN user u ON p.id_user = u.id WHERE p.id = %s", (prediksi_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        flash("Data tidak ditemukan.", "error")
        return redirect(url_for('riwayat'))

    hasil = json.loads(row['jumlah'])

    # Urutkan hasil berdasarkan tahun dan bulan
    from calendar import month_name
    month_map = {name: i for i, name in enumerate(month_name) if name}
    ordered = dict(sorted(hasil.items(), key=lambda x: (int(x[0].split()[1]), month_map[x[0].split()[0]])))

    # Peta nama bulan ke Bahasa Indonesia
    bulan_id_map = {
        "January": "Januari", "February": "Februari", "March": "Maret",
        "April": "April", "May": "Mei", "June": "Juni", "July": "Juli",
        "August": "Agustus", "September": "September", "October": "Oktober",
        "November": "November", "December": "Desember"
    }

    # Buat rentang prediksi dengan Bahasa Indonesia
    if ordered:
        bulan_pertama, tahun_pertama = list(ordered.keys())[0].split()
        bulan_terakhir, tahun_terakhir = list(ordered.keys())[-1].split()
        rentang = f"{bulan_id_map[bulan_pertama]} {tahun_pertama} - {bulan_id_map[bulan_terakhir]} {tahun_terakhir}"
        row['rentang'] = rentang
    else:
        row['rentang'] = "-"

    # Siapkan data chart
    chart_data = {
        "labels": list(ordered.keys()),
        "data": list(ordered.values())
    }

    return render_template(
        'detail_prediksi.html',
        data=row,
        hasil=ordered,
        chart_data=json.dumps(chart_data)
    )



@app.route('/hapus_prediksi/<int:prediksi_id>')
def hapus_prediksi(prediksi_id):
    try:
        conn = koneksi_mysql()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediksi WHERE id = %s", (prediksi_id,))
        conn.commit()
        conn.close()

        # Jika tidak ada baris yang terhapus (misalnya ID tidak ditemukan)
        if cursor.rowcount == 0:
            flash("Data tidak ditemukan atau sudah dihapus.", "error")
        else:
            flash("Data berhasil dihapus.", "success")
    except Exception as e:
        flash(f"Gagal menghapus data: {str(e)}", "error")
    return redirect(url_for('riwayat'))



@app.route('/export/excel/<int:prediksi_id>')
def export_excel(prediksi_id):
    conn = koneksi_mysql()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM prediksi WHERE id = %s", (prediksi_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return "Data tidak ditemukan", 404

    hasil = json.loads(row['jumlah'])

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Prediksi"
    ws.append(["Bulan", "Jumlah"])
    for bulan, jumlah in hasil.items():
        ws.append([bulan, jumlah])

    stream = BytesIO()
    wb.save(stream)
    stream.seek(0)

    return send_file(stream, download_name=f"prediksi_{prediksi_id}.xlsx", as_attachment=True, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ===================== 🔁 JALANKAN APP =====================
if __name__ == '__main__':
    app.run(debug=True)
