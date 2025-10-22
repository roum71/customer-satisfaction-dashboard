#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard — v7.6 (Secure + Gauges + Colored Services + Pareto 80%)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
from pathlib import Path

# =========================================================
# 🔐 Users and Roles
# =========================================================
USER_KEYS = {
    "Public Services Department": {"password": "psd2025", "role": "center", "file": "Center_Public_Services.csv"},
    "Ras Al Khaimah Municipality": {"password": "rakm2025", "role": "center", "file": "Center_RAK_Municipality.csv"},
    "Sheikh Saud Center-Ras Al Khaimah Courts": {"password": "ssc2025", "role": "center", "file": "Center_Sheikh_Saud_Courts.csv"},
    "Sheikh Saqr Center-Ras Al Khaimah Courts": {"password": "ssq2025", "role": "center", "file": "Center_Sheikh_Saqr_Courts.csv"},
    "Executive Council": {"password": "admin2025", "role": "admin", "file": None},
}

# =========================================================
# 🎨 Page Setup
# =========================================================
st.set_page_config(page_title="لوحة مؤشرات رضا المتعاملين — الإصدار 7.6", layout="wide")
PASTEL = px.colors.qualitative.Pastel

# =========================================================
# 🌐 Language
# =========================================================
lang = st.sidebar.radio("🌍 اللغة / Language", ["العربية", "English"], index=0)
if lang == "العربية":
    st.markdown("""
        <style>
        html, body, [class*="css"] {direction:rtl;text-align:right;font-family:"Tajawal","Cairo","Segoe UI";}
        </style>
    """, unsafe_allow_html=True)

# =========================================================
# 🔑 Login
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
    st.sidebar.subheader("🔒 كلمة المرور / Password")
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
# 📥 Load Data
# =========================================================
if role == "admin":
    st.markdown("### 🏛️ وضع الأمانة العامة")
    target_center = st.selectbox("اختر المركز:", ["All Centers (Master)"] + [c for c in USER_KEYS if c != "Executive Council"])
    file_path = "Centers_Master.csv" if target_center == "All Centers (Master)" else USER_KEYS[target_center]["file"]
else:
    file_path = USER_KEYS[center]["file"]
    st.markdown(f"### 📊 لوحة مركز {center}")
    st.info("📂 يتم تحميل البيانات الخاصة بالمركز فقط.")

try:
    df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
except Exception as e:
    st.error(f"❌ تعذر تحميل الملف: {e}")
    st.stop()

# =========================================================
# 📗 Lookup Merge
# =========================================================
lookup_catalog = {}
lookup_path = Path("Data_tables.xlsx")
if lookup_path.exists():
    xls = pd.ExcelFile(lookup_path)
    for sheet in xls.sheet_names:
        tbl = pd.read_excel(xls, sheet_name=sheet)
        tbl.columns = [c.strip().upper() for c in tbl.columns]
        lookup_catalog[sheet.upper()] = tbl
    for col in df.columns:
        if col.upper() in lookup_catalog:
            tbl = lookup_catalog[col.upper()]
            merge_key = "CODE" if "CODE" in tbl.columns else tbl.columns[0]
            lang_col = "ARABIC" if lang == "العربية" else "ENGLISH"
            if lang_col in tbl.columns:
                df = df.merge(tbl[[merge_key, lang_col]], how="left", left_on=col, right_on=merge_key)
                df.rename(columns={lang_col: f"{col}_name"}, inplace=True)
                df.drop(columns=[merge_key], inplace=True, errors="ignore")

# =========================================================
# Helper functions
# =========================================================
def series_to_percent(vals: pd.Series) -> float:
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if len(vals)==0: return np.nan
    mx = vals.max()
    if mx <= 5: return ((vals - 1)/4*100).mean()
    elif mx <= 10: return ((vals - 1)/9*100).mean()
    else: return vals.mean()

def detect_nps(df):
    cands = [c for c in df.columns if "nps" in c.lower() or "recommend" in c.lower()]
    if not cands: return np.nan
    s = pd.to_numeric(df[cands[0]], errors="coerce").dropna()
    if len(s)==0: return np.nan
    promoters = (s >= 9).sum(); detractors = (s <= 6).sum()
    return (promoters - detractors)/len(s)*100

# =========================================================
# Tabs
# =========================================================
tab_kpis, tab_services, tab_pareto = st.tabs(["📊 المؤشرات","📋 الخدمات","💬 Pareto"])

# =========================================================
# KPIs with Gauges
# =========================================================
with tab_kpis:
    st.subheader("📊 المؤشرات الرئيسية (CSAT / CES / NPS)")

    csat = series_to_percent(df.get("Dim6.1", pd.Series(dtype=float)))
    ces = series_to_percent(df.get("Dim6.2", pd.Series(dtype=float)))
    nps = detect_nps(df)

    col1, col2, col3 = st.columns(3)
    for (col, val, label) in zip([col1, col2, col3], [csat, ces, nps], ["CSAT", "CES", "NPS"]):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val if not np.isnan(val) else 0,
            title={'text': label},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2ecc71"},
                'steps': [
                    {'range': [0, 60], 'color': '#f5b7b1'},
                    {'range': [60, 80], 'color': '#fcf3cf'},
                    {'range': [80, 100], 'color': '#c8f7c5'}
                ]
            }
        ))
        col.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📋 Services Tab (Color-coded by CSAT)
# =========================================================
with tab_services:
    st.subheader("📋 تحليل حسب الخدمة")

    if "SERVICE_name" in df.columns:
        # حساب المؤشرات لكل خدمة
        service_data = df.groupby("SERVICE_name").agg({
            "Dim6.1": series_to_percent,
            "Dim6.2": series_to_percent,
            next((c for c in df.columns if "nps" in c.lower() or "recommend" in c.lower()), None): detect_nps
        }).reset_index()

        service_data.rename(columns={"Dim6.1": "CSAT", "Dim6.2": "CES"}, inplace=True)
        counts = df["SERVICE_name"].value_counts().reset_index()
        counts.columns = ["SERVICE_name", "Count"]
        service_data = counts.merge(service_data, on="SERVICE_name", how="left")

        def highlight_csat(val):
            if pd.isna(val):
                return "background-color: white;"
            elif val >= 80:
                return "background-color: #c8f7c5;"  # أخضر فاتح
            elif val < 60:
                return "background-color: #f5b7b1;"  # أحمر فاتح
            else:
                return "background-color: #fcf3cf;"  # أصفر فاتح

        st.dataframe(
            service_data.style.applymap(highlight_csat, subset=["CSAT"]).format({
                "CSAT": "{:.1f}",
                "CES": "{:.1f}",
                "Count": "{:,.0f}"
            })
        )
        st.caption("🟩 CSAT ≥ 80 أداء ممتاز | 🟨 بين 60 و80 متوسط | 🟥 أقل من 60 ضعيف")
    else:
        st.info("⚠️ لا يوجد حقل SERVICE_name في البيانات.")

# =========================================================
# 💬 Pareto (80% Red)
# =========================================================
with tab_pareto:
    st.subheader("💬 تحليل نصوص الشكاوى (Pareto)")
    text_cols = [c for c in df.columns if any(x in c.lower() for x in ["comment","ملاحظ","شكوى","reason","unsat"])]
    if text_cols:
        text_col = text_cols[0]
        df["__clean"] = df[text_col].astype(str).str.lower().replace(r"[^\u0600-\u06FFA-Za-z0-9\s]","",regex=True)
        df = df[~df["__clean"].isin(["","لا يوجد","none","no","nothing"])]

        themes = {
            "Waiting / الانتظار":["انتظار","delay","بطء"],
            "Staff / الموظفون":["موظف","staff","تعامل"],
            "Fees / الرسوم":["رسوم","fee","cost"],
            "Process / الإجراءات":["اجراء","process","انجاز"],
            "Service / الخدمة":["خدم","service","جودة"],
            "Platform / المنصة":["تطبيق","app","website","system"],
        }

        def classify(t):
            for th,words in themes.items():
                for w in words:
                    if w in t: return th
            return "Other / أخرى"

        df["Theme"] = df["__clean"].apply(classify)
        df = df[df["Theme"] != "Other / أخرى"]
        theme_counts = df["Theme"].value_counts().reset_index()
        theme_counts.columns = ["Theme","Count"]
        theme_counts["%"] = theme_counts["Count"]/theme_counts["Count"].sum()*100
        theme_counts["Cum%"] = theme_counts["%"].cumsum()

        theme_counts["Color"] = np.where(theme_counts["Cum%"] <= 80, "#e74c3c", "#95a5a6")

        fig = go.Figure()
        fig.add_bar(x=theme_counts["Theme"], y=theme_counts["Count"], marker_color=theme_counts["Color"], name="Count")
        fig.add_scatter(x=theme_counts["Theme"], y=theme_counts["Cum%"], name="Cumulative %", yaxis="y2", mode="lines+markers")
        fig.update_layout(yaxis=dict(title="Count"), yaxis2=dict(title="Cum%", overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ لا يوجد عمود نصي لتحليل Pareto.")
