#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard — v8.8 (Final Stable)
✅ Fix persistent NameError (target_center)
✅ Stable across all user roles and Streamlit reruns
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
    "Executive Council": {"password": "admin2025", "role": "admin", "file": "Centers_Master.csv"},
}

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="لوحة تجربة المتعاملين مراكز رأس الخيمة — الإصدار 1.0", layout="wide")
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
params = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
center_from_link = params.get("center", [None])[0]
center_options = list(USER_KEYS.keys())

if center_from_link and center_from_link in USER_KEYS:
    selected_center = center_from_link
else:
    st.sidebar.header("🏢 اختر المركز / Select Center")
    selected_center = st.sidebar.selectbox("Select Center / اختر المركز", center_options)

if "authorized" not in st.session_state:
    st.session_state.update({
        "authorized": False, "center": None, "role": None, "target_center": None
    })

if not st.session_state["authorized"] or st.session_state["center"] != selected_center:
    st.sidebar.subheader("🔑 كلمة المرور / Password")
    password = st.sidebar.text_input("Password", type="password")
    if password == USER_KEYS[selected_center]["password"]:
        st.session_state.update({
            "authorized": True,
            "center": selected_center,
            "role": USER_KEYS[selected_center]["role"],
            "file": USER_KEYS[selected_center]["file"],
            "target_center": None
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
        if Path(file).exists():
            return pd.read_csv(file, encoding="utf-8", low_memory=False)
        return None
    except Exception:
        return None

# ✅ تعريف target_center داخل session دائمًا
if role == "admin":
    st.markdown("### 🏛️ الأمانة العامة")
    st.session_state["target_center"] = st.selectbox(
        "اختر المركز:",
        ["All Centers (Master)"] + [c for c in USER_KEYS if c != "Executive Council"]
    )
    target_center = st.session_state["target_center"]
    file_path = "Centers_Master.csv" if target_center == "All Centers (Master)" else USER_KEYS[target_center]["file"]
else:
    target_center = st.session_state.get("target_center", center)
    file_path = USER_KEYS[center]["file"]
    st.markdown(f"### 📊 لوحة مركز {center}")

df = safe_read(file_path)
if df is None or df.empty:
    st.error(f"❌ لا يمكن تحميل الملف: {file_path}")
    st.stop()

# =========================================================
# FUNCTIONS
# =========================================================
def series_to_percent(vals):
    s = pd.to_numeric(vals, errors="coerce").dropna()
    if s.empty:
        return np.nan
    mx = s.max()
    if mx <= 5:
        return ((s - 1) / 4 * 100).mean()
    elif mx <= 10:
        return ((s - 1) / 9 * 100).mean()
    else:
        return s.mean()

def detect_nps(df_in):
    cands = [c for c in df_in.columns if ("nps" in c.lower()) or ("recommend" in c.lower())]
    if not cands:
        return np.nan
    s = pd.to_numeric(df_in[cands[0]], errors="coerce").dropna()
    if s.empty:
        return np.nan
    promoters = (s >= 9).sum()
    detractors = (s <= 6).sum()
    return (promoters - detractors) / len(s) * 100

# =========================================================
# FILTERS
# =========================================================
filter_cols = [c for c in df.columns if c.endswith("_name") and c.upper() in ["GENDER_NAME","SERVICE_NAME","SECTOR_NAME","NATIONALITY_NAME","CENTER_NAME"]]
filters = {}
with st.sidebar.expander("🎛️ الفلاتر / Filters"):
    for col in filter_cols:
        options = df[col].dropna().unique().tolist()
        selection = st.multiselect(col.replace("_name",""), options, default=options)
        filters[col] = selection
for col, values in filters.items():
    df = df[df[col].isin(values)]

# =========================================================
# TABS
# =========================================================
tab_data, tab_sample, tab_kpis, tab_services, tab_pareto = st.tabs([
    "📁 البيانات","📈 توزيع العينة","📊 المؤشرات","📋 الخدمات","💬 Pareto"
])

# =========================================================
# 📈 SAMPLE TAB
# =========================================================
with tab_sample:
    st.subheader("📈 توزيع العينة")
    total = len(df)
    st.markdown(f"### 🧮 إجمالي الردود: {total:,}")
    if total == 0:
        st.info("لا توجد بيانات.")
        st.stop()

    chart_type = st.radio("📊 نوع الرسم", ["دائري Pie", "أعمدة Bar"], index=0, horizontal=True)
    grouping = ["CENTER_name"] if "CENTER_name" in df.columns else []

    if role == "admin" and target_center == "All Centers (Master)" and grouping:
        summary = df.groupby(grouping).size().reset_index(name="Count")
        fig = px.bar(summary, x="CENTER_name", y="Count", color="CENTER_name",
                     title="عدد الردود حسب المركز", color_discrete_sequence=PASTEL)
        st.plotly_chart(fig, use_container_width=True)

    for col in filter_cols:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "Count"]
        counts["%"] = counts["Count"] / total * 100
        title = f"{col.replace('_name','')} — {total:,} رد"
        if chart_type == "دائري Pie":
            fig = px.pie(counts, names=col, values="Count", hole=0.3,
                         title=title, color_discrete_sequence=PASTEL)
        else:
            fig = px.bar(counts, x=col, y="Count", text="Count", title=title,
                         color=col, color_discrete_sequence=PASTEL)
            fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📊 KPIs TAB (All Centers merge)
# =========================================================
with tab_kpis:
    st.subheader("📊 المؤشرات الرئيسية (CSAT / CES / NPS)")

    if role == "admin" and target_center == "All Centers (Master)":
        combined = []
        for c, info in USER_KEYS.items():
            if c == "Executive Council": continue
            if Path(info["file"]).exists():
                dfc = pd.read_csv(info["file"], encoding="utf-8", low_memory=False)
                dfc["Center"] = c
                combined.append(dfc)
        if combined:
            df_all = pd.concat(combined, ignore_index=True)
            summary = df_all.groupby("Center").agg(
                CSAT=("Dim6.1", series_to_percent),
                CES=("Dim6.2", series_to_percent),
                NPS=("Center", lambda x: detect_nps(df_all[df_all["Center"] == x.name]))
            ).reset_index()
            st.dataframe(summary.style.format({"CSAT": "{:.1f}", "CES": "{:.1f}", "NPS": "{:.1f}"}))
            fig = px.bar(summary.melt(id_vars="Center", value_vars=["CSAT", "CES", "NPS"]),
                         x="Center", y="value", color="variable",
                         barmode="group", title="مقارنة المؤشرات بين المراكز",
                         color_discrete_sequence=PASTEL)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ لا توجد بيانات كافية لجميع المراكز.")
    else:
        csat = series_to_percent(df.get("Dim6.1", pd.Series(dtype=float)))
        ces = series_to_percent(df.get("Dim6.2", pd.Series(dtype=float)))
        nps = detect_nps(df)
        c1, c2, c3 = st.columns(3)
        for col, val, name in zip([c1, c2, c3], [csat, ces, nps], ["CSAT", "CES", "NPS"]):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val if not np.isnan(val) else 0,
                title={'text': name},
                gauge={'axis': {'range': [0, 100]},
                       'steps': [
                           {'range': [0, 60], 'color': '#f5b7b1'},
                           {'range': [60, 80], 'color': '#fcf3cf'},
                           {'range': [80, 100], 'color': '#c8f7c5'}],
                       'bar': {'color': '#2ecc71'}}))
            col.plotly_chart(fig, use_container_width=True)

# =========================================================
# 💬 PARETO TAB (merged for All Centers)
# =========================================================
with tab_pareto:
    st.subheader("💬 تحليل نصوص الملاحظات (Pareto)")

    if role == "admin" and target_center == "All Centers (Master)":
        combined = []
        for c, info in USER_KEYS.items():
            if c == "Executive Council": continue
            if Path(info["file"]).exists():
                dfc = pd.read_csv(info["file"], encoding="utf-8", low_memory=False)
                combined.append(dfc)
        if combined:
            df = pd.concat(combined, ignore_index=True)

    text_cols = [c for c in df.columns if any(k in c.lower() for k in ["unsat","comment","ملاحظ","reason"])]
    if not text_cols:
        st.warning("⚠️ لا يوجد عمود نصي.")
        st.stop()
    col = text_cols[0]
    df["__clean"] = df[col].astype(str).str.lower()
    df["__clean"] = df["__clean"].replace(r"[^\u0600-\u06FFA-Za-z0-9\s]", " ", regex=True)
    df["__clean"] = df["__clean"].replace(r"\s+", " ", regex=True).str.strip()
    empty = {"", " ", "لا يوجد", "لايوجد", "لا شيء", "no", "none", "nothing", "جيد", "ممتاز", "ok"}
    df = df[~df["__clean"].isin(empty)]
    df = df[df["__clean"].apply(lambda x: len(x.split()) >= 3)]

    themes = {
        "Parking / مواقف السيارات": ["موقف","مواقف","parking","السيارات"],
        "Waiting / الانتظار": ["انتظار","بطء","delay","slow"],
        "Staff / الموظفون": ["موظف","تعامل","staff"],
        "Fees / الرسوم": ["رسوم","دفع","fee"],
        "Process / الإجراءات": ["اجراء","process","انجاز"],
        "Platform / المنصة": ["تطبيق","app","system","website"],
        "Facility / المكان": ["مكان","نظافة","ازدحام"],
        "Communication / التواصل": ["رد","تواصل","اتصال"]
    }

    def classify_theme(t):
        for th, ws in themes.items():
            if any(w in t for w in ws): return th
        return "Other / أخرى"

    df["Theme"] = df["__clean"].apply(classify_theme)
    df = df[df["Theme"] != "Other / أخرى"]
    counts = df["Theme"].value_counts().reset_index()
    counts.columns = ["Theme","Count"]
    counts["%"] = (counts["Count"]/counts["Count"].sum()*100).round(1)
    counts["Cum%"] = counts["%"].cumsum()
    counts["Color"] = np.where(counts["Cum%"]<=80,"#e74c3c","#95a5a6")

    st.dataframe(counts.style.format({"%":"{:.1f}","Cum%":"{:.1f}"}))
    fig = go.Figure()
    fig.add_bar(x=counts["Theme"], y=counts["Count"], marker_color=counts["Color"])
    fig.add_scatter(x=counts["Theme"], y=counts["Cum%"], name="Cumulative %", yaxis="y2", mode="lines+markers")
    fig.update_layout(title="Pareto — المحاور الرئيسية",
                      yaxis=dict(title="عدد الملاحظات"),
                      yaxis2=dict(title="النسبة التراكمية (%)",overlaying="y",side="right"))
    st.plotly_chart(fig, use_container_width=True)
