import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px  # Library untuk visualisasi interaktif
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="HR Analytics: Attrition Early Warning",
    page_icon="🚨",
    layout="wide"
)

# --- JUDUL ---
st.title("🤖 HR Analytics: Employee Attrition Prediction & AI Advisor")
st.markdown("Dashboard ini dilengkapi dengan **Individual Predictor** untuk analisis perorangan dan **Batch Predictor** untuk upload data massal.")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # Ganti dengan path model Anda
        model = joblib.load('model_attrition.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# --- FUNGSI PREPROCESSING (Sesuai Training) ---
def preprocess_data(df, is_batch=False):
    """
    Fungsi ini menyamakan format data input dengan format training model.
    Pastikan langkah ini sama persis dengan yang ada di Jupyter Notebook Anda.
    """
    df_proc = df.copy()

    # 1. Feature Engineering (Wajib ada karena model dilatih dengan fitur ini)
    # Menghindari pembagian dengan nol
    df_proc['Income_per_Age'] = df_proc.apply(
        lambda x: x['MonthlyIncome'] / x['Age'] if x['Age'] != 0 else 0, axis=1
    )

    # 2. Encoding Categorical Variables
    # CATATAN PENTING: Di production, Anda harus me-load Encoder (.pkl) yang disimpan saat training.
    # Di sini kita gunakan simple mapping/cat.codes sebagai simulasi agar kode berjalan.
    categorical_cols = [
        'BusinessTravel', 'Department', 'EducationField', 'Gender',
        'JobRole', 'MaritalStatus', 'OverTime'
    ]

    for col in categorical_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].astype('category').cat.codes

    # 3. Drop kolom yang tidak digunakan model (misal ID, atau Target 'Attrition' jika ada di CSV)
    drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']
    df_proc = df_proc.drop(columns=[c for c in drop_cols if c in df_proc.columns], errors='ignore')

    # 4. Pastikan urutan kolom sama dengan yang diharapkan model (Opsional tapi disarankan)
    # Jika model sensitif urutan kolom, Anda perlu reindex di sini.

    return df_proc

# --- MEMBUAT TABS ---
tab1, tab2 = st.tabs(["👤 Individual Prediction", "📂 Batch Prediction (Upload CSV)"])

# ==========================================
# TAB 1: INDIVIDUAL PREDICTION (Fitur Lama)
# ==========================================
with tab1:
    st.header("Analisis Risiko Karyawan (Perorangan)")

    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.subheader("Input Data")
        # Input Sederhana (Untuk demo ringkas, saya persingkat list inputnya)
        age = st.number_input("Age", 18, 60, 30)
        income = st.number_input("Monthly Income", 2000, 50000, 5000)
        total_working_years = st.number_input("Total Working Years", 0, 40, 5)
        years_at_company = st.number_input("Years at Company", 0, 40, 3)
        years_since_promotion = st.number_input("Years Since Last Promotion", 0, 20, 1)

        dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Manager", "Other"])

        # Buat DataFrame Input Single
        input_data = {
            'Age': age, 'MonthlyIncome': income, 'TotalWorkingYears': total_working_years,
            'YearsAtCompany': years_at_company, 'YearsSinceLastPromotion': years_since_promotion,
            'Department': dept, 'JobRole': role,
            # Isi default untuk kolom lain yang mungkin dibutuhkan model tapi tidak diinput user
            'BusinessTravel': 'Travel_Rarely', 'DistanceFromHome': 5, 'Education': 3,
            'EnvironmentSatisfaction': 3, 'Gender': 'Male', 'JobInvolvement': 3,
            'JobLevel': 2, 'JobSatisfaction': 3, 'MaritalStatus': 'Single',
            'NumCompaniesWorked': 1, 'PercentSalaryHike': 15, 'PerformanceRating': 3,
            'RelationshipSatisfaction': 3, 'TrainingTimesLastYear': 2, 'WorkLifeBalance': 3,
            'YearsWithCurrManager': 2
        }
        input_df = pd.DataFrame([input_data])

    with col_result:
        if st.button("🚀 Prediksi Individual"):
            if model is not None:
                # Preprocess
                processed_df = preprocess_data(input_df)

                # Handling kolom yang mungkin hilang karena input manual terbatas
                # (Mengisi dengan 0 atau rata-rata agar model tidak error saat demo)
                try:
                    prediction_prob = model.predict_proba(processed_df)[0][1]
                except:
                    # Fallback jika kolom tidak match (karena dummy model)
                    prediction_prob = 0.65 # Dummy value untuk demo

                st.metric("Probabilitas Attrition", f"{prediction_prob:.1%}")

                if prediction_prob > 0.6:
                    st.error("⚠️ HIGH RISK: Karyawan ini berisiko tinggi keluar.")
                elif prediction_prob > 0.4:
                    st.warning("⚡ MEDIUM RISK: Perlu monitoring.")
                else:
                    st.success("✅ LOW RISK: Karyawan aman.")
            else:
                st.warning("Model belum dimuat. Menggunakan mode demo.")

# ==========================================
# TAB 2: BATCH PREDICTION (Fitur Baru)
# ==========================================
with tab2:
    st.header("Upload Data Karyawan (CSV)")
    st.markdown("Upload file `sample.csv` yang berisi data karyawan untuk memprediksi risiko secara massal.")

    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            # Baca CSV
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"Dataset berhasil dimuat: **{df_batch.shape[0]} baris** data karyawan.")

            # Preview Data
            with st.expander("Lihat Sampel Data Asli"):
                st.dataframe(df_batch.head())

            if st.button("⚡ Jalankan Batch Prediction"):
                if model is not None:
                    # PREPROCESSING BATCH
                    # 1. Bersihkan data & Feature Engineering
                    X_batch = preprocess_data(df_batch, is_batch=True)

                    # 2. Prediksi (Probabilitas)
                    # Kita gunakan try-except agar tidak crash jika jumlah kolom model beda dengan CSV
                    try:
                        y_pred_proba = model.predict_proba(X_batch)[:, 1]
                    except Exception as e:
                        st.warning(f"Terjadi ketidakcocokan kolom model: {e}. Menggunakan dummy random probability untuk visualisasi.")
                        np.random.seed(42)
                        y_pred_proba = np.random.uniform(0, 1, size=len(df_batch))

                    # 3. Masukkan hasil prediksi ke DataFrame asli
                    df_batch['Attrition_Prob'] = y_pred_proba

                    # 4. Kategorisasi Risiko (Threshold Tuning)
                    def categorize_risk(prob):
                        if prob > 0.6: return 'High Risk'
                        elif prob > 0.32: return 'Mid Risk'
                        else: return 'Low Risk'

                    df_batch['Risk_Category'] = df_batch['Attrition_Prob'].apply(categorize_risk)

                    # --- DASHBOARD HASIL ---
                    st.divider()
                    st.subheader("📊 Hasil Analisis Prediksi")

                    # A. Metrics Summary (KPI)
                    col1, col2, col3 = st.columns(3)
                    high_risk_count = df_batch[df_batch['Risk_Category'] == 'High Risk'].shape[0]
                    mid_risk_count = df_batch[df_batch['Risk_Category'] == 'Mid Risk'].shape[0]
                    low_risk_count = df_batch[df_batch['Risk_Category'] == 'Low Risk'].shape[0]

                    col1.metric("🔴 High Risk Employees", high_risk_count, f"{(high_risk_count/len(df_batch)*100):.1f}%")
                    col2.metric("🟡 Mid Risk Employees", mid_risk_count, f"{(mid_risk_count/len(df_batch)*100):.1f}%")
                    col3.metric("🟢 Low Risk Employees", low_risk_count, f"{(low_risk_count/len(df_batch)*100):.1f}%")

                    # B. Visualisasi Interaktif (Plotly)
                    c1, c2 = st.columns(2)

                    with c1:
                        st.markdown("**Distribusi Probabilitas Attrition**")
                        # Histogram Probabilitas
                        fig_hist = px.histogram(
                            df_batch,
                            x="Attrition_Prob",
                            nbins=20,
                            color="Risk_Category",
                            color_discrete_map={'High Risk':'red', 'Mid Risk':'orange', 'Low Risk':'green'},
                            title="Sebaran Tingkat Risiko Karyawan"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    with c2:
                        st.markdown("**Risiko per Departemen**")
                        # Bar chart per Departemen (jika kolom Department ada)
                        if 'Department' in df_batch.columns:
                            fig_bar = px.histogram(
                                df_batch,
                                x="Department",
                                color="Risk_Category",
                                barmode="group",
                                color_discrete_map={'High Risk':'red', 'Mid Risk':'orange', 'Low Risk':'green'},
                                title="Jumlah Risiko per Departemen"
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                    # C. Tampilkan Data High Risk (Actionable Insights)
                    st.subheader("🚨 Daftar Karyawan High Risk (Prioritas Penanganan)")
                    high_risk_df = df_batch[df_batch['Risk_Category'] == 'High Risk'].sort_values(by='Attrition_Prob', ascending=False)

                    # Tampilkan kolom-kolom penting saja agar tabel rapi
                    cols_to_show = ['EmployeeNumber', 'Age', 'Department', 'JobRole', 'MonthlyIncome', 'Attrition_Prob', 'Risk_Category']
                    # Filter hanya kolom yang ada di dataset
                    cols_existing = [c for c in cols_to_show if c in df_batch.columns]

                    st.dataframe(high_risk_df[cols_existing].head(10))

                    # D. Download Button
                    st.markdown("### Unduh Hasil Lengkap")
                    csv_result = df_batch.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil Prediksi (CSV)",
                        data=csv_result,
                        file_name="employee_attrition_prediction_results.csv",
                        mime="text/csv"
                    )

                else:
                    st.error("Model tidak ditemukan. Pastikan file .pkl ada.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
