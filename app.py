#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard — v7.4.4 Secure Advanced Edition
- Full analytics (Sample, KPIs, Dimensions, NPS, Pareto)
- Secure login by center
- Excel export per center or all centers for admin
"""

# =========================================================
# 📦 Import Libraries
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re, io, zipfile
from datetime import datetime
from pathlib import Path

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
st.set_page_config(page_title="لوحة مؤشرات رضا المتعاملين — الإصدار 7.4.4", layout="wide")
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
# 🧠 Helper Functions
# =========================================================
def series_to_percent(vals: pd.Series) -> float:
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    mx = vals.max()
    if mx <= 5:
        return ((vals - 1) / 4 * 100).mean()
    elif mx <= 10:
        return ((vals - 1) / 9 * 100).mean()
    else:
        return vals.mean()

def detect_nps(df):
    cands = [c for c in df.columns if "nps" in c.lower() or "recommend" in c.lower()]
    if not cands:
        return np.nan
    s = pd.to_numeric(df[cands[0]], errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    promoters = (s >= 9).sum()
    detractors = (s <= 6).sum()
    return (promoters - detractors) / len(s) * 100

# =========================================================
# 🧭 Tabs
# =========================================================
tab_sample, tab_kpis, tab_dims, tab_nps, tab_pareto = st.tabs(
    ["📈 توزيع العينة", "📊 المؤشرات", "📉 الأبعاد", "🎯 NPS", "💬 Pareto"]
)

# =========================================================
# 📈 Sample Tab
# =========================================================
with tab_sample:
    st.subheader("📈 توزيع العينة")
    lookup_cols = [c for c in df.columns if any(k in c.lower() for k in ["gender", "sector", "center", "nationality", "service"])]
    for col in lookup_cols:
        if df[col].nunique() > 1:
            fig = px.pie(df, names=col, title=f"توزيع {col}", color_discrete_sequence=PASTEL)
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📊 KPIs Tab
# =========================================================
with tab_kpis:
    st.subheader("📊 المؤشرات الرئيسية (CSAT / CES / NPS)")
    csat = series_to_percent(df.select_dtypes(include=np.number).mean(axis=1))
    ces = series_to_percent(df.select_dtypes(include=np.number).median(axis=1))
    nps = detect_nps(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("😊 CSAT (%)", f"{csat:.2f}" if not np.isnan(csat) else "N/A")
    c2.metric("⭐ CES (%)", f"{ces:.2f}" if not np.isnan(ces) else "N/A")
    c3.metric("🎯 NPS", f"{nps:.2f}" if not np.isnan(nps) else "N/A")

# =========================================================
# 📉 Dimensions Tab
# =========================================================
with tab_dims:
    st.subheader("📉 الأبعاد")
    dim_cols = [c for c in df.columns if re.match(r"Dim[1-6]\.[0-9]+", str(c))]
    dims_scores = {}
    for i in range(1, 6):
        items = [c for c in dim_cols if str(c).startswith(f"Dim{i}.")]
        if items:
            vals = df[items].apply(pd.to_numeric, errors="coerce").stack().dropna()
            dims_scores[f"Dim{i}"] = series_to_percent(vals)
    if dims_scores:
        ddf = pd.DataFrame(list(dims_scores.items()), columns=["Dimension", "Score"])
        fig = px.bar(ddf, x="Dimension", y="Score", text_auto=".1f", color="Dimension", color_discrete_sequence=PASTEL)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("لم يتم العثور على أعمدة Dim1–Dim6.")

# =========================================================
# 🎯 NPS Tab
# =========================================================
with tab_nps:
    st.subheader("🎯 صافي نقاط الترويج (NPS)")
    nps_cols = [c for c in df.columns if "nps" in c.lower() or "recommend" in c.lower()]
    if nps_cols:
        s = pd.to_numeric(df[nps_cols[0]], errors="coerce").dropna()
        nps_buckets = pd.cut(s, bins=[0, 6, 8, 10], labels=["Detractor", "Passive", "Promoter"])
        pie_df = nps_buckets.value_counts().reset_index()
        pie_df.columns = ["Type", "Count"]
        fig = px.pie(pie_df, names="Type", values="Count",
                     color="Type", color_discrete_map={"Promoter": "#2ecc71", "Passive": "#95a5a6", "Detractor": "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⚠️ لا يوجد عمود NPS في البيانات.")

# =========================================================
# 💬 Pareto Tab
# =========================================================
with tab_pareto:
    st.subheader("💬 تحليل نصوص الشكاوى (Pareto)")
    text_cols = [c for c in df.columns if any(x in c.lower() for x in ["most_unsat", "comment", "ملاحظ", "شكوى", "reason"])]
    if text_cols:
        text_col = text_cols[0]
        df["__clean"] = df[text_col].astype(str).str.lower().replace(r"[^\u0600-\u06FFA-Za-z0-9\s]", "", regex=True)
        df = df[~df["__clean"].isin(["", "لا يوجد", "none", "no", "nothing"])]

        themes = {
            "Waiting / الانتظار": ["انتظار", "delay", "بطء"],
            "Staff / الموظفون": ["موظف", "staff", "تعامل"],
            "Fees / الرسوم": ["رسوم", "fee", "cost"],
            "Process / الإجراءات": ["اجراء", "process", "انجاز"],
            "Service / الخدمة": ["خدم", "service", "جودة"],
            "Platform / المنصة": ["تطبيق", "app", "website", "system"],
        }

        def classify(text):
            for th, words in themes.items():
                for w in words:
                    if w in text:
                        return th
            return "Other / أخرى"

        df["Theme"] = df["__clean"].apply(classify)
        df = df[df["Theme"] != "Other / أخرى"]
        theme_counts = df["Theme"].value_counts().reset_index()
        theme_counts.columns = ["Theme", "Count"]
        theme_counts["%"] = theme_counts["Count"] / theme_counts["Count"].sum() * 100
        theme_counts["Cum%"] = theme_counts["%"].cumsum()

        st.dataframe(theme_counts)
        fig = go.Figure()
        fig.add_bar(x=theme_counts["Theme"], y=theme_counts["Count"], name="Count")
        fig.add_scatter(x=theme_counts["Theme"], y=theme_counts["Cum%"], name="Cumulative %", yaxis="y2")
        fig.update_layout(yaxis=dict(title="Count"), yaxis2=dict(title="Cum%", overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)

        # === Excel Export ===
        if st.button("⬇️ تنزيل تقرير Excel"):
            ts = datetime.now().strftime("%Y-%m-%d")
            out_name = f"Report_{center.replace(' ', '_')}_{ts}.xlsx"
            with pd.ExcelWriter(out_name, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="RawData")
                theme_counts.to_excel(writer, index=False, sheet_name="Pareto")
            with open(out_name, "rb") as f:
                st.download_button("📥 تحميل التقرير", data=f.read(), file_name=out_name,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.warning("⚠️ لا يوجد عمود نصي لتحليل Pareto.")
