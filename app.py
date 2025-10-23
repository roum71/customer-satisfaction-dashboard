#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard — v10.8 (Filters Names Edition)
Unified | Secure | Multi-Center | Lookup | KPI Gauges | Dimensions | Pareto | Services Overview
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
# 🔄 تحويل الأكواد إلى أسماء باستخدام Lookup
# =========================================================
def translate_column(df, col_name):
    sheet = col_name.upper()
    if sheet in lookup_catalog:
        tbl = lookup_catalog[sheet]
        code_col = next((c for c in tbl.columns if "CODE" in c or "VALUE" in c), None)
        ar_col = next((c for c in tbl.columns if "ARABIC" in c), None)
        en_col = next((c for c in tbl.columns if "ENGLISH" in c), None)
        if code_col and ar_col and en_col:
            mapping = dict(zip(tbl[code_col], tbl[ar_col if lang == "العربية" else en_col]))
            df[col_name + "_NAME"] = df[col_name].map(mapping)
            return col_name + "_NAME"
    return col_name

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
    if not cands: return np.nan, 0, 0, 0
    s = pd.to_numeric(df[cands[0]], errors="coerce").dropna()
    if len(s)==0: return np.nan, 0, 0, 0
    promoters = (s>=9).sum()
    passives = ((s>=7)&(s<=8)).sum()
    detractors = (s<=6).sum()
    total = len(s)
    promoters_pct = promoters/total*100
    passives_pct = passives/total*100
    detractors_pct = detractors/total*100
    nps = promoters_pct - detractors_pct
    return nps, promoters_pct, passives_pct, detractors_pct

# =========================================================
# FILTERS
# =========================================================
filter_cols = [c for c in df.columns if any(k in c.upper() for k in ["GENDER", "SERVICE", "SECTOR", "NATIONALITY", "CENTER"])]
filters = {}
with st.sidebar.expander("🎛️ الفلاتر / Filters"):
    for col in filter_cols:
        col_translated = translate_column(df, col)
        options = df[col_translated].dropna().unique().tolist()
        selection = st.multiselect(col, options, default=options)
        filters[col_translated] = selection
for col, values in filters.items():
    df = df[df[col].isin(values)]

# =========================================================
# تبويبات التحليل
# =========================================================
tab_data, tab_sample, tab_kpis, tab_dimensions, tab_services, tab_pareto = st.tabs(
    ["📁 البيانات", "📈 توزيع العينة", "📊 المؤشرات", "🧩 الأبعاد", "📋 الخدمات", "💬 Pareto"]
)

# =========================================================
# 📈 SAMPLE TAB — الفلاتر بالأسماء
# =========================================================
with tab_sample:
    st.subheader("📈 توزيع العينة")
    total = len(df)
    st.markdown(f"### 🧮 إجمالي الردود: {total:,}")
    chart_type = st.radio("📊 نوع الرسم", ["دائري Pie", "أعمدة Bar"], index=0, horizontal=True)
    for col in filters.keys():
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "Count"]
        counts["%"] = counts["Count"]/total*100
        title = f"{col} — {total:,} رد"
        if chart_type == "دائري Pie":
            fig = px.pie(counts, names=col, values="Count", hole=0.3, title=title, color_discrete_sequence=PASTEL)
        else:
            fig = px.bar(counts, x=col, y="Count", text="Count", color=col, color_discrete_sequence=PASTEL)
        st.plotly_chart(fig, use_container_width=True)
