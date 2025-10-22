#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard — v7.7 (Full Secure + Gauges + Lookup + Filters + Pareto 80%)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from pathlib import Path

# =========================================================
# 🔐 USERS & ACCESS CONTROL
# =========================================================
USER_KEYS = {
    "Public Services Department": {"password": "psd2025", "role": "center", "file": "Center_Public_Services.csv"},
    "Ras Al Khaimah Municipality": {"password": "rakm2025", "role": "center", "file": "Center_RAK_Municipality.csv"},
    "Sheikh Saud Center-Ras Al Khaimah Courts": {"password": "ssc2025", "role": "center", "file": "Center_Sheikh_Saud_Courts.csv"},
    "Sheikh Saqr Center-Ras Al Khaimah Courts": {"password": "ssq2025", "role": "center", "file": "Center_Sheikh_Saqr_Courts.csv"},
    "Executive Council": {"password": "admin2025", "role": "admin", "file": None},
}

# =========================================================
# 🎨 PAGE CONFIG
# =========================================================
st.set_page_config(page_title="لوحة مؤشرات رضا المتعاملين — الإصدار 7.7", layout="wide")
PASTEL = px.colors.qualitative.Pastel

# =========================================================
# 🌍 LANGUAGE SELECTION
# =========================================================
lang = st.sidebar.radio("🌍 اللغة / Language", ["العربية", "English"], index=0)
if lang == "العربية":
    st.markdown("""
        <style>
        html, body, [class*="css"] {direction:rtl;text-align:right;font-family:"Tajawal","Cairo","Segoe UI";}
        </style>
    """, unsafe_allow_html=True)

# =========================================================
# 🔑 LOGIN
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
# 📥 LOAD DATA
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
# 📗 LOOKUP TABLES
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
# 🧮 HELPER FUNCTIONS
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
# 🧩 FILTERS
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
# 📊 TABS
# =========================================================
tab_sample, tab_kpis, tab_services, tab_pareto = st.tabs(["📈 توزيع العينة","📊 المؤشرات","📋 الخدمات","💬 Pareto"])

# =========================================================
# 📈 SAMPLE DISTRIBUTION
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
        title = f"{col.replace('_name','')} — {total:,} رد"

        if chart_type == "دائري Pie":
            fig = px.pie(counts, names=col, values="Count", hole=0.3, title=title, color_discrete_sequence=PASTEL)
            fig.update_traces(text=counts["Count"], textinfo="value+label")
        else:
            fig = px.bar(counts, x=col, y="Count", text="Count", title=title, color=col, color_discrete_sequence=PASTEL)
            fig.update_traces(textposition="outside")

        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📊 KPIs GAUGES
# =========================================================
with tab_kpis:
    st.subheader("📊 المؤشرات الرئيسية (CSAT / CES / NPS)")
    csat = series_to_percent(df.get("Dim6.1", pd.Series(dtype=float)))
    ces = series_to_percent(df.get("Dim6.2", pd.Series(dtype=float)))
    nps = detect_nps(df)

    col1, col2, col3 = st.columns(3)
    for col, val, name in zip([col1,col2,col3],[csat,ces,nps],["CSAT","CES","NPS"]):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val if not np.isnan(val) else 0,
            title={'text': name},
            gauge={
                'axis': {'range':[0,100]},
                'steps': [
                    {'range':[0,60],'color':'#f5b7b1'},
                    {'range':[60,80],'color':'#fcf3cf'},
                    {'range':[80,100],'color':'#c8f7c5'}
                ],
                'bar': {'color':'#2ecc71'}
            }
        ))
        col.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📋 SERVICES TAB (CSAT COLOR)
# =========================================================
with tab_services:
    st.subheader("📋 تحليل حسب الخدمة")
    if "SERVICE_name" in df.columns:
        # حساب المؤشرات
        service_data = df.groupby("SERVICE_name").agg({
            "Dim6.1": series_to_percent,
            "Dim6.2": series_to_percent
        }).reset_index()
        service_data.rename(columns={"Dim6.1":"CSAT","Dim6.2":"CES"}, inplace=True)

        # حساب NPS يدويًا
        nps_col = next((c for c in df.columns if "nps" in c.lower() or "recommend" in c.lower()), None)
        if nps_col:
            nps_vals = []
            for sname, group in df.groupby("SERVICE_name"):
                s = pd.to_numeric(group[nps_col], errors="coerce").dropna()
                if len(s)>0:
                    promoters = (s>=9).sum(); detractors = (s<=6).sum()
                    score = (promoters - detractors)/len(s)*100
                else:
                    score = np.nan
                nps_vals.append({"SERVICE_name": sname, "NPS": score})
            nps_df = pd.DataFrame(nps_vals)
            service_data = service_data.merge(nps_df, on="SERVICE_name", how="left")

        # دمج عدد الردود
        counts = df["SERVICE_name"].value_counts().reset_index()
        counts.columns = ["SERVICE_name","Count"]
        service_data = counts.merge(service_data, on="SERVICE_name", how="left")

        # تلوين CSAT
        def highlight(val):
            if pd.isna(val): return "background-color:white;"
            if val>=80: return "background-color:#c8f7c5;"
            elif val<60: return "background-color:#f5b7b1;"
            else: return "background-color:#fcf3cf;"
        st.dataframe(service_data.style.applymap(highlight, subset=["CSAT"]).format({"CSAT":"{:.1f}","CES":"{:.1f}","NPS":"{:.1f}"}))
        st.caption("🟩 CSAT ≥80 ممتاز | 🟨 60–80 متوسط | 🟥 <60 منخفض")
    else:
        st.warning("⚠️ لا يوجد عمود SERVICE_name في البيانات.")

# =========================================================
# 💬 PARETO ANALYSIS
# =========================================================
with tab_pareto:
    st.subheader("💬 تحليل نصوص الملاحظات (Pareto)")
    text_cols = [c for c in df.columns if any(k in c.lower() for k in ["comment","ملاحظ","شكوى","reason","unsat"])]
    if text_cols:
        col = text_cols[0]
        df["__clean"] = df[col].astype(str).str.lower().replace(r"[^\u0600-\u06FFA-Za-z0-9\s]","",regex=True)
        df = df[~df["__clean"].isin(["","لا يوجد","none","no","nothing"])]

        themes = {
            "Waiting / الانتظار":["انتظار","delay","بطء"],
            "Staff / الموظفون":["موظف","staff","تعامل"],
            "Fees / الرسوم":["رسوم","fee","cost"],
            "Process / الإجراءات":["اجراء","process","انجاز"],
            "Service / الخدمة":["خدم","service","جودة"],
            "Platform / المنصة":["تطبيق","app","website","system"]
        }

        def classify(t):
            for th, ws in themes.items():
                if any(w in t for w in ws): return th
            return "Other / أخرى"

        df["Theme"] = df["__clean"].apply(classify)
        df = df[df["Theme"]!="Other / أخرى"]
        counts = df["Theme"].value_counts().reset_index()
        counts.columns = ["Theme","Count"]
        counts["%"] = counts["Count"]/counts["Count"].sum()*100
        counts["Cum%"] = counts["%"].cumsum()
        counts["Color"] = np.where(counts["Cum%"]<=80,"#e74c3c","#95a5a6")

        fig = go.Figure()
        fig.add_bar(x=counts["Theme"], y=counts["Count"], marker_color=counts["Color"], name="Count")
        fig.add_scatter(x=counts["Theme"], y=counts["Cum%"], name="Cumulative %", yaxis="y2", mode="lines+markers")
        fig.update_layout(yaxis=dict(title="Count"), yaxis2=dict(title="Cum%", overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ لا يوجد عمود نصي لتحليل Pareto.")
