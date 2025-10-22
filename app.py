#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard — v7.4.4 (Unified Secure Edition)
Auto login per center via credentials
Admin (Executive Council) can access all centers or master file
Includes dimensions, KPIs, NPS, Pareto charts
"""

# =========================================================
# 📚 Import Libraries
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import re

# =========================================================
# 🔐 Users and Roles
# =========================================================
USER_KEYS = {
    "Public Services Department": {
        "password": "psd2025",
        "role": "center",
        "file": "Center_Public_Services.csv",
    },
    "Ras Al Khaimah Municipality": {
        "password": "rakm2025",
        "role": "center",
        "file": "Center_RAK_Municipality.csv",
    },
    "Sheikh Saud Center-Ras Al Khaimah Courts": {
        "password": "ssc2025",
        "role": "center",
        "file": "Center_Sheikh_Saud_Courts.csv",
    },
    "Sheikh Saqr Center-Ras Al Khaimah Courts": {
        "password": "ssq2025",
        "role": "center",
        "file": "Center_Sheikh_Saqr_Courts.csv",
    },
    "Executive Council": {
        "password": "admin2025",
        "role": "admin",
        "file": None,
    },
}

# =========================================================
# 🎨 Page Setup
# =========================================================
st.set_page_config(page_title="لوحة مؤشرات رضا المتعاملين 7.4.4 (خفيفة)", layout="wide")
PASTEL = px.colors.qualitative.Pastel

# =========================================================
# 🌐 Language
# =========================================================
lang = st.sidebar.radio("🌍 اللغة / Language", ["العربية", "English"], index=0)
rtl = True if lang == "العربية" else False

if rtl:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            direction: rtl;
            text-align: right;
            font-family: "Tajawal", "Cairo", "Segoe UI", sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 🧩 Center Selection
# =========================================================
params = st.query_params
center_from_link = params.get("center", [None])[0]
center_options = list(USER_KEYS.keys())

if center_from_link and center_from_link in USER_KEYS:
    selected_center = center_from_link
else:
    st.sidebar.header("🏢 اختر المركز / Select Center")
    selected_center = st.sidebar.selectbox("Select Center / اختر المركز", center_options)

# =========================================================
# 🔑 Login
# =========================================================
if "authorized" not in st.session_state:
    st.session_state["authorized"] = False
if "center" not in st.session_state:
    st.session_state["center"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None

if not st.session_state["authorized"] or st.session_state["center"] != selected_center:
    st.sidebar.subheader("🔒 كلمة المرور / Password")
    password = st.sidebar.text_input("Password", type="password")

    if password == USER_KEYS[selected_center]["password"]:
        st.session_state["authorized"] = True
        st.session_state["center"] = selected_center
        st.session_state["role"] = USER_KEYS[selected_center]["role"]
        st.session_state["file"] = USER_KEYS[selected_center]["file"]
        st.success(f"✅ تم تسجيل الدخول بنجاح كمركز: {selected_center}")
        st.rerun()
    elif password:
        st.error("🚫 كلمة المرور غير صحيحة.")
        st.stop()
    else:
        st.warning("يرجى إدخال كلمة المرور.")
        st.stop()

# =========================================================
# 📁 Load Data Automatically
# =========================================================
center = st.session_state["center"]
role = st.session_state["role"]

if role == "admin":
    st.markdown("### 🏛️ وضع الأمانة العامة (Admin Mode)")
    target_center = st.selectbox(
        "اختر المركز:",
        ["All Centers (Master)"] + [c for c in USER_KEYS.keys() if c != "Executive Council"],
    )
    file_path = (
        "Centers_Master.csv"
        if target_center == "All Centers (Master)"
        else USER_KEYS[target_center]["file"]
    )
else:
    file_path = USER_KEYS[center]["file"]
    st.markdown(f"### 📊 لوحة مركز {center}")
    st.info("📂 يتم تحميل البيانات تلقائيًا من الملف المرتبط بالمركز.")

try:
    df = pd.read_csv(file_path, encoding="utf-8")
    st.success(f"✅ تم تحميل البيانات ({len(df)} صفًا).")
except Exception as e:
    st.error(f"❌ تعذر تحميل الملف: {e}")
    st.stop()

# =========================================================
# 🧠 كشف الأعمدة المساعدة (Lookup)
# =========================================================
lookup_cols = [c for c in df.columns if any(k in c.lower() for k in ["gender", "sector", "center", "nationality"])]
st.success(f"lookup: {', '.join(lookup_cols)}")

# =========================================================
# 📊 توزيع العينة
# =========================================================
st.markdown("### 📊 توزيع العينة")
for col in lookup_cols:
    if df[col].nunique() > 1:
        fig = px.histogram(
            df, x=col, color=col, title=f"توزيع {col}", color_discrete_sequence=PASTEL
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📈 مؤشرات NPS / CSAT
# =========================================================
st.markdown("### 🌟 مؤشرات الأداء NPS / CSAT")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if numeric_cols:
    metrics = df[numeric_cols].mean().round(2)
    col1, col2, col3 = st.columns(3)
    col1.metric("📈 متوسط NPS", metrics.get("NPS", np.nan))
    col2.metric("😊 متوسط CSAT", metrics.get("CSAT", np.nan))
    col3.metric("⭐ متوسط CES", metrics.get("CES", np.nan))
else:
    st.warning("⚠️ لا توجد أعمدة رقمية لعرض مؤشرات الأداء.")

# =========================================================
# 📉 تحليل Pareto للشكاوى / النصوص
# =========================================================
st.markdown("### 📊 تحليل نصوص الشكاوى (Pareto)")
text_cols = [c for c in df.columns if df[c].dtype == object and "comment" in c.lower() or "text" in c.lower()]
if text_cols:
    text_col = text_cols[0]
    df["Theme"] = df[text_col].fillna("غير محدد")
    pareto_df = df["Theme"].value_counts().reset_index()
    pareto_df.columns = ["Theme", "Count"]
    pareto_df["Cum%"] = pareto_df["Count"].cumsum() / pareto_df["Count"].sum() * 100

    fig = go.Figure()
    fig.add_bar(x=pareto_df["Theme"], y=pareto_df["Count"], name="العدد")
    fig.add_scatter(
        x=pareto_df["Theme"],
        y=pareto_df["Cum%"],
        mode="lines+markers",
        name="النسبة التراكمية %",
        yaxis="y2",
    )

    fig.update_layout(
        title="تحليل باريتو للنصوص",
        yaxis=dict(title="العدد"),
        yaxis2=dict(title="النسبة التراكمية", overlaying="y", side="right"),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ لم يتم العثور على عمود نصي للتحليل (Comments/Text).")

# =========================================================
# ✅ Summary
# =========================================================
st.markdown("✅ تم تحليل البيانات بنجاح. جميع الأقسام جاهزة للعرض.")
