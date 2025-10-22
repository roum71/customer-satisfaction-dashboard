#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard — v7.4.4 (Secure Intelligent Tabs Edition)
Unified version with full login system + smart KPI detection + dynamic tabs
"""

# =========================================================
# 📦 Import Libraries
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
# 🎨 Page Configuration
# =========================================================
st.set_page_config(page_title="لوحة مؤشرات رضا المتعاملين (الإصدار 7.4.4)", layout="wide")
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
# 🏢 Login Section
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
# 📁 Load Data
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
# 🔍 كشف الأعمدة
# =========================================================
lookup_cols = [c for c in df.columns if any(k in c.lower() for k in ["gender", "sector", "center", "nationality"])]
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# =========================================================
# 🧠 Smart KPI Detection
# =========================================================
def detect_csat(df):
    candidates = [c for c in df.columns if re.search(r"q\d+", c.lower())]
    data = df[candidates].select_dtypes(include=np.number)
    return data.mean(axis=1).mean() * 20 if not data.empty else np.nan

def detect_ces(df):
    candidates = [c for c in df.columns if re.search(r"ease|effort|time", c.lower())]
    data = df[candidates].select_dtypes(include=np.number)
    return data.mean(axis=1).mean() * 14.28 if not data.empty else np.nan

def detect_nps(df):
    candidates = [c for c in df.columns if re.search(r"nps|recommend", c.lower())]
    if not candidates:
        return np.nan
    s = df[candidates[0]].dropna()
    promoters = (s >= 9).sum()
    detractors = (s <= 6).sum()
    return ((promoters - detractors) / len(s)) * 100 if len(s) > 0 else np.nan

csat_score = round(detect_csat(df), 2)
ces_score = round(detect_ces(df), 2)
nps_score = round(detect_nps(df), 2)

# =========================================================
# 🧭 Tabs Navigation
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 العينة", "📊 المؤشرات", "📈 الأبعاد", "⭐ NPS", "🧩 Pareto"]
)

# =========================================================
# 📋 Tab 1: Sample Distribution
# =========================================================
with tab1:
    st.subheader("📋 توزيع العينة")
    for col in lookup_cols:
        if df[col].nunique() > 1:
            fig = px.histogram(df, x=col, color=col, color_discrete_sequence=PASTEL)
            fig.update_layout(title=f"توزيع {col}")
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📊 Tab 2: KPIs
# =========================================================
with tab2:
    st.subheader("📊 مؤشرات الأداء (CSAT / CES / NPS)")
    col1, col2, col3 = st.columns(3)
    col1.metric("😊 CSAT", f"{csat_score:.2f}" if not np.isnan(csat_score) else "N/A")
    col2.metric("⭐ CES", f"{ces_score:.2f}" if not np.isnan(ces_score) else "N/A")
    col3.metric("📈 NPS", f"{nps_score:.2f}" if not np.isnan(nps_score) else "N/A")

# =========================================================
# 📈 Tab 3: Dimensions
# =========================================================
with tab3:
    st.subheader("📈 تحليل الأبعاد / Dimensions")
    dim_cols = [c for c in df.columns if re.search(r"dim|aspect|factor", c.lower())]
    if dim_cols:
        dim_mean = df[dim_cols].mean().reset_index()
        dim_mean.columns = ["Dimension", "Score"]
        fig = px.bar(dim_mean, x="Dimension", y="Score", color="Score", color_continuous_scale="teal")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ لا توجد أعمدة للأبعاد في هذا الملف.")

# =========================================================
# ⭐ Tab 4: NPS Distribution
# =========================================================
with tab4:
    st.subheader("⭐ توزيع NPS")
    candidates = [c for c in df.columns if re.search(r"nps|recommend", c.lower())]
    if candidates:
        fig = px.histogram(df, x=candidates[0], nbins=10, color_discrete_sequence=PASTEL)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ لم يتم العثور على عمود NPS.")

# =========================================================
# 🧩 Tab 5: Pareto
# =========================================================
with tab5:
    st.subheader("🧩 تحليل نصوص الشكاوى (Pareto)")
    text_cols = [
        c for c in df.columns if any(x in c.lower() for x in ["comment", "text", "note", "remark", "ملاحظ", "شكوى"])
    ]
    if text_cols:
        text_col = text_cols[0]
        df["Theme"] = df[text_col].fillna("غير محدد")
        pareto_df = df["Theme"].value_counts().reset_index()
        pareto_df.columns = ["Theme", "Count"]
        pareto_df["Cum%"] = pareto_df["Count"].cumsum() / pareto_df["Count"].sum() * 100

        fig = go.Figure()
        fig.add_bar(x=pareto_df["Theme"], y=pareto_df["Count"], name="العدد")
        fig.add_scatter(x=pareto_df["Theme"], y=pareto_df["Cum%"], mode="lines+markers", name="النسبة التراكمية %", yaxis="y2")
        fig.update_layout(
            yaxis=dict(title="العدد"),
            yaxis2=dict(title="النسبة التراكمية", overlaying="y", side="right"),
            title="تحليل Pareto للنصوص المفتوحة",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ لم يتم العثور على عمود نصي للتحليل.")

# =========================================================
# ✅ Summary
# =========================================================
st.success("✅ تم تحليل البيانات بنجاح — جميع التبويبات جاهزة للعرض.")
