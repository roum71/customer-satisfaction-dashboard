#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard — v10.7 (Executive Edition)
Unified | Secure | Multi-Center | Lookup | KPI Gauges | Pareto | Services Overview
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, re
from datetime import datetime
from pathlib import Path

# =========================================================
# 🔐 USERS
# =========================================================
USER_KEYS = {
    "Public Services Department": {"password": "psd2025", "role": "center", "file": "Center_Public_Services.csv"},
    "Ras Al Khaimah Municipality": {"password": "rakm2025", "role": "center", "file": "Center_RAK_Municipality.csv"},
    "Sheikh Saud Center-Ras Al Khaimah Courts": {"password": "ssc2025", "role": "center", "file": "Center_Sheikh_Saud_Courts.csv"},
    "Sheikh Saqr Center-Ras Al Khaimah Courts": {"password": "ssq2025", "role": "center", "file": "Center_Sheikh_Saqr_Courts.csv"},
    "Executive Council": {"password": "admin2025", "role": "admin", "file": "Centers_Master.csv"}
}

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="لوحة تجربة المتعاملين — رأس الخيمة", layout="wide")
PASTEL = px.colors.qualitative.Pastel

# =========================================================
# LANGUAGE
# =========================================================
lang = st.sidebar.radio("🌍 اللغة / Language", ["العربية", "English"], index=0)
if lang == "العربية":
    st.markdown("""
        <style>
        html, body, [class*="css"] {direction:rtl;text-align:right;font-family:"Tajawal","Cairo","Segoe UI";}
        </style>
    """, unsafe_allow_html=True)

# =========================================================
# LOGIN
# =========================================================
params = st.query_params
center_from_link = params.get("center", [None])[0]
center_options = list(USER_KEYS.keys())

if center_from_link and center_from_link in USER_KEYS:
    selected_center = center_from_link
else:
    st.sidebar.header("🏢 اختر المركز / Select Center")
    selected_center = st.sidebar.selectbox("Select Center / اختر المركز", center_options)

if "authorized" not in st.session_state:
    st.session_state.update({"authorized": False, "center": None, "role": None})

if not st.session_state["authorized"] or st.session_state["center"] != selected_center:
    st.sidebar.subheader("🔑 كلمة المرور / Password")
    password = st.sidebar.text_input("Password", type="password")
    if password == USER_KEYS[selected_center]["password"]:
        st.session_state.update({
            "authorized": True,
            "center": selected_center,
            "role": USER_KEYS[selected_center]["role"],
            "file": USER_KEYS[selected_center]["file"]
        })
        st.success(f"✅ تم تسجيل الدخول كمركز: {selected_center}")
        st.rerun()
    elif password:
        st.error("🚫 كلمة المرور غير صحيحة.")
        st.stop()
    else:
        st.warning("يرجى إدخال كلمة المرور.")
        st.stop()

center, role = st.session_state["center"], st.session_state["role"]

# =========================================================
# LOAD DATA
# =========================================================
def safe_read(file):
    try:
        return pd.read_csv(file, encoding="utf-8", low_memory=False)
    except Exception:
        return None

file_path = USER_KEYS[center]["file"]
df = safe_read(file_path)
if df is None:
    st.error(f"❌ تعذر تحميل الملف: {file_path}")
    st.stop()

# =========================================================
# LOOKUP TABLES
# =========================================================
lookup_path = Path("Data_tables.xlsx")
lookup_catalog = {}
if lookup_path.exists():
    xls = pd.ExcelFile(lookup_path)
    for sheet in xls.sheet_names:
        tbl = pd.read_excel(xls, sheet_name=sheet)
        tbl.columns = [c.strip().upper() for c in tbl.columns]
        lookup_catalog[sheet.upper()] = tbl

# =========================================================
# UTILS
# =========================================================
def series_to_percent(vals):
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    mx = vals.max()
    if mx <= 5: return ((vals - 1)/4*100).mean()
    elif mx <= 10: return ((vals - 1)/9*100).mean()
    else: return vals.mean()

def detect_nps(df):
    cands = [c for c in df.columns if "nps" in c.lower() or "recommend" in c.lower()]
    if not cands: return np.nan
    s = pd.to_numeric(df[cands[0]], errors="coerce").dropna()
    if len(s)==0: return np.nan
    promoters = (s>=9).sum(); detractors = (s<=6).sum()
    return (promoters - detractors)/len(s)*100

# =========================================================
# FILTERS
# =========================================================
filter_cols = [c for c in df.columns if any(k in c.upper() for k in ["GENDER", "SERVICE", "SECTOR", "NATIONALITY", "CENTER"])]
filters = {}
with st.sidebar.expander("🎛️ الفلاتر / Filters"):
    for col in filter_cols:
        options = df[col].dropna().unique().tolist()
        selection = st.multiselect(col, options, default=options)
        filters[col] = selection
for col, values in filters.items():
    df = df[df[col].isin(values)]

# =========================================================
# TABS
# =========================================================
tab_data, tab_sample, tab_kpis, tab_services, tab_pareto = st.tabs(
    ["البيانات", "📈 توزيع العينة", "📊 المؤشرات", "📋 الخدمات", "💬 Pareto"]
)

# =========================================================
# 📁 DATA TAB — Multi-language headers
# =========================================================
with tab_data:
    st.subheader("📁 البيانات بعد الفلاتر")

    questions_map_ar, questions_map_en = {}, {}
    if "QUESTIONS" in lookup_catalog:
        qtbl = lookup_catalog["QUESTIONS"]
        qtbl.columns = [c.strip().upper() for c in qtbl.columns]
        code_col = next((c for c in qtbl.columns if "CODE" in c or "DIMENSION" in c), None)
        ar_col = next((c for c in qtbl.columns if "ARABIC" in c or c == "ARABIC"), None)
        en_col = next((c for c in qtbl.columns if "ENGLISH" in c or c == "ENGLISH"), None)

        if code_col and ar_col and en_col:
            qtbl["CODE_NORM"] = qtbl[code_col].astype(str).str.strip().str.upper()
            questions_map_ar = dict(zip(qtbl["CODE_NORM"], qtbl[ar_col]))
            questions_map_en = dict(zip(qtbl["CODE_NORM"], qtbl[en_col]))

    df_display = df.copy()
    df_display.columns = [c.strip() for c in df_display.columns]
    ar_row = [questions_map_ar.get(c.strip().upper(), "") for c in df_display.columns]
    en_row = [questions_map_en.get(c.strip().upper(), "") for c in df_display.columns]
    df_final = pd.concat([pd.DataFrame([ar_row, en_row], columns=df_display.columns), df_display], ignore_index=True)

    st.dataframe(df_final, use_container_width=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_final.to_excel(writer, index=False)
    st.download_button("📥 تنزيل البيانات", buffer.getvalue(), file_name=f"Filtered_Data_{ts}.xlsx")

# =========================================================
# 📈 SAMPLE TAB
# =========================================================
with tab_sample:
    st.subheader("📈 توزيع العينة")
    total = len(df)
    st.markdown(f"### 🧮 إجمالي الردود: {total:,}")
    chart_type = st.radio("📊 نوع الرسم", ["دائري Pie", "أعمدة Bar"], index=0, horizontal=True)
    for col in filter_cols:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "Count"]
        counts["%"] = counts["Count"]/total*100
        title = f"{col} — {total:,} رد"
        if chart_type == "دائري Pie":
            fig = px.pie(counts, names=col, values="Count", hole=0.3, title=title, color_discrete_sequence=PASTEL)
        else:
            fig = px.bar(counts, x=col, y="Count", text="Count", color=col, color_discrete_sequence=PASTEL)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📊 KPIs TAB
# =========================================================
with tab_kpis:
    st.subheader("📊 المؤشرات الرئيسية (CSAT / CES / NPS)")
    csat = series_to_percent(df.get("Dim6.1", pd.Series(dtype=float)))
    ces = series_to_percent(df.get("Dim6.2", pd.Series(dtype=float)))
    nps = detect_nps(df)
    for val, name in zip([csat, ces, nps], ["CSAT", "CES", "NPS"]):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val if not np.isnan(val) else 0,
            title={'text': name},
            gauge={'axis': {'range': [0, 100]},
                   'steps': [{'range': [0, 60], 'color': '#f5b7b1'},
                             {'range': [60, 80], 'color': '#fcf3cf'},
                             {'range': [80, 100], 'color': '#c8f7c5'}],
                   'bar': {'color': '#2ecc71'}}))
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📋 SERVICES TAB
# =========================================================
with tab_services:
    st.subheader("📋 تحليل الخدمات")

    service_col = next((c for c in df.columns if "SERVICE" in c.upper()), None)
    if not service_col:
        st.warning("⚠️ لا يوجد عمود خدمة في البيانات.")
    else:
        if "SERVICE" in lookup_catalog:
            st.info("📘 تم استخدام ملف الترجمة لعرض أسماء الخدمات.")
            st_df = lookup_catalog["SERVICE"]
            st_df.columns = [c.strip().upper() for c in st_df.columns]
            code_col = next((c for c in st_df.columns if "CODE" in c or "SERVICE" in c), None)
            ar_col = next((c for c in st_df.columns if "ARABIC" in c), None)
            en_col = next((c for c in st_df.columns if "ENGLISH" in c), None)
            if code_col and ar_col and en_col:
                name_map = dict(zip(st_df[code_col].astype(str).str.strip(), st_df[ar_col if lang=="العربية" else en_col]))
                df[service_col] = df[service_col].astype(str).replace(name_map)

        summary = df.groupby(service_col).agg({
            "Dim6.1": series_to_percent,
            "Dim6.2": series_to_percent
        }).reset_index().rename(columns={"Dim6.1": "CSAT", "Dim6.2": "CES"})
        summary["عدد الردود"] = df.groupby(service_col).size().values
        summary["التصنيف اللوني"] = np.where(summary["CSAT"] >= 80, "مرتفع",
                                     np.where(summary["CSAT"] >= 60, "متوسط", "منخفض"))

        st.dataframe(summary, use_container_width=True)
        fig = px.bar(summary.sort_values("CSAT", ascending=False),
                     x=service_col, y="CSAT", color="التصنيف اللوني",
                     text="عدد الردود", color_discrete_sequence=PASTEL,
                     title="رضا المتعاملين حسب الخدمة")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 💬 PARETO TAB
# =========================================================
with tab_pareto:
    st.subheader("💬 تحليل الملاحظات (Pareto)")
    text_cols=[c for c in df.columns if any(k in c.lower() for k in ["comment","ملاحظ","unsat","reason"])]
    if not text_cols:
        st.warning("⚠️ لا يوجد عمود نصي لتحليل Pareto.")
    else:
        col=text_cols[0]
        df["__clean"]=df[col].astype(str).str.lower()
        df["__clean"]=df["__clean"].replace(r"[^\u0600-\u06FFA-Za-z0-9\s]"," ",regex=True)
        df["__clean"]=df["__clean"].replace(r"\s+"," ",regex=True).str.strip()
        empty_terms={""," ","لا يوجد","لايوجد","لا شيء","no","none","nothing","جيد","ممتاز","ok"}
        df=df[~df["__clean"].isin(empty_terms)]
        df=df[df["__clean"].apply(lambda x: len(x.split())>=3)]

        themes={
            "Parking / مواقف السيارات":["موقف","مواقف","parking"],
            "Waiting / الانتظار":["انتظار","بطء","delay","slow"],
            "Staff / الموظفون":["موظف","تعامل","staff"],
            "Fees / الرسوم":["رسوم","دفع","fee"],
            "Process / الإجراءات":["اجراء","process","انجاز"],
            "Platform / المنصة":["تطبيق","app","system"],
            "Facility / المكان":["مكان","نظافة","ازدحام"],
            "Communication / التواصل":["رد","تواصل","اتصال"]
        }

        def classify_theme(t):
            for th,ws in themes.items():
                if any(w in t for w in ws): return th
            return "Other / أخرى"

        df["Theme"]=df["__clean"].apply(classify_theme)
        counts=df["Theme"].value_counts().reset_index()
        counts.columns=["Theme","Count"]
        counts["%"]=counts["Count"]/counts["Count"].sum()*100
        counts["Cum%"]=counts["%"].cumsum()

        fig=go.Figure()
        fig.add_bar(x=counts["Theme"],y=counts["Count"],name="عدد الملاحظات",marker_color="#5dade2")
        fig.add_scatter(x=counts["Theme"],y=counts["Cum%"],name="النسبة التراكمية",yaxis="y2",mode="lines+markers")
        fig.update_layout(yaxis2=dict(overlaying="y",side="right",title="النسبة التراكمية (%)"),
                          title="Pareto — أهم المحاور",bargap=0.2)
        st.plotly_chart(fig,use_container_width=True)
