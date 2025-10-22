#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard — v7.9 (Full Secure + Lookup + Gauges + Smart Filters + Sorted Services + Pareto AI Themes)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from pathlib import Path
import io
from datetime import datetime

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
st.set_page_config(page_title="لوحة مؤشرات رضا المتعاملين — الإصدار 7.9", layout="wide")
PASTEL = px.colors.qualitative.Pastel

# =========================================================
# 🌍 LANGUAGE
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
# 🧮 FUNCTIONS
# =========================================================
def series_to_percent(vals: pd.Series) -> float:
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if len(vals) == 0: return np.nan
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
# 🎛️ FILTERS
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
# 📈 SAMPLE
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
# 📊 KPIs
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
            gauge={'axis':{'range':[0,100]},
                   'steps':[{'range':[0,60],'color':'#f5b7b1'},
                            {'range':[60,80],'color':'#fcf3cf'},
                            {'range':[80,100],'color':'#c8f7c5'}],
                   'bar':{'color':'#2ecc71'}}))
        col.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📋 SERVICES TAB (Sorted + Export)
# =========================================================
with tab_services:
    st.subheader("📋 تحليل حسب الخدمة")

    if "SERVICE_name" in df.columns:
        service_data = df.groupby("SERVICE_name").agg({
            "Dim6.1": series_to_percent,
            "Dim6.2": series_to_percent
        }).reset_index()
        service_data.rename(columns={"Dim6.1":"CSAT","Dim6.2":"CES"}, inplace=True)

        nps_col = next((c for c in df.columns if "nps" in c.lower() or "recommend" in c.lower()), None)
        if nps_col:
            nps_vals=[]
            for sname, g in df.groupby("SERVICE_name"):
                s=pd.to_numeric(g[nps_col],errors="coerce").dropna()
                if len(s)>0:
                    promoters=(s>=9).sum(); detractors=(s<=6).sum()
                    score=(promoters-detractors)/len(s)*100
                else: score=np.nan
                nps_vals.append({"SERVICE_name":sname,"NPS":score})
            service_data=service_data.merge(pd.DataFrame(nps_vals),on="SERVICE_name",how="left")

        counts=df["SERVICE_name"].value_counts().reset_index()
        counts.columns=["SERVICE_name","Count"]
        service_data=counts.merge(service_data,on="SERVICE_name",how="left")
        service_data.sort_values(by="CSAT",ascending=False,inplace=True)

        def highlight(val):
            if pd.isna(val): return "background-color:white;"
            if val>=80: return "background-color:#c8f7c5;"
            elif val<60: return "background-color:#f5b7b1;"
            else: return "background-color:#fcf3cf;"

        st.dataframe(service_data.style.applymap(highlight,subset=["CSAT"]).format({"CSAT":"{:.1f}","CES":"{:.1f}","NPS":"{:.1f}","Count":"{:,.0f}"}))
        st.caption("🟩 ≥80 ممتاز | 🟨 60–80 متوسط | 🟥 <60 ضعيف")

        # 📥 EXPORT CLEAN DATA
        clean_cols=[c for c in df.columns if c.endswith("_name") or c.lower().startswith("dim") or c.lower() in ["nps","center","service","gender","sector","nationality"]]
        df_clean=df[clean_cols].copy()
        df_clean.columns=[c.replace("_name","").capitalize() for c in df_clean.columns]

        buffer=io.BytesIO()
        ts=datetime.now().strftime("%Y-%m-%d_%H%M")
        with pd.ExcelWriter(buffer,engine="openpyxl") as writer:
            df_clean.to_excel(writer,index=False,sheet_name="Clean_Data")

        st.download_button("📥 تنزيل البيانات النظيفة (Excel)",data=buffer.getvalue(),
                           file_name=f"Clean_Customer_Data_{ts}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.warning("⚠️ لا يوجد عمود SERVICE_name في البيانات.")

# =========================================================
# 💬 PARETO ANALYSIS (Smart Themes + Filters)
# =========================================================
with tab_pareto:
    st.subheader("💬 تحليل نصوص الملاحظات (Pareto المحاور الفعلية)")

    text_cols=[c for c in df.columns if any(k in c.lower() for k in ["most_unsat","comment","ملاحظ","reason"])]
    if text_cols:
        col=text_cols[0]
        df["__clean"]=df[col].astype(str).str.lower()
        df["__clean"]=df["__clean"].replace(r"[^\u0600-\u06FFA-Za-z0-9\s]"," ",regex=True)
        df["__clean"]=df["__clean"].replace(r"\s+"," ",regex=True).str.strip()

        empty_terms={""," ","لا يوجد","لايوجد","لا يوجد شيء","لا شي","لا شيء","مافي","ما في","ماشي","لا أعلم","no","none","nothing","nothing to say","good","ok","fine","ممتاز","جيد","تمام"}
        df=df[~df["__clean"].isin(empty_terms)]
        df=df[df["__clean"].apply(lambda x: len(x.split())>=3)]

        themes={
            "Parking / مواقف السيارات":["موقف","مواقف","باركن","parking","السيارات"],
            "Waiting / الانتظار":["انتظار","بطء","delay","slow","long"],
            "Staff / الموظفون":["موظف","موظفين","تعامل","المعاملة","staff"],
            "Fees / الرسوم":["رسوم","دفع","fee","payment","expensive","cost"],
            "Process / الإجراءات":["اجراء","الإجراءات","انجاز","process","steps"],
            "Platform / المنصة":["تطبيق","app","system","portal","website","النظام","الرد","support"],
            "Facility / المكان":["مكان","قسم","المكاتب","نظافة","ازدحام","facility"],
            "Appointments / المواعيد":["موعد","schedule","time","booking"],
            "Communication / التواصل":["رد","تواصل","اتصال","call","response"],
            "Availability / التوفر":["لا يوجد","عدم","no","none","nothing","توفر"]
        }

        def classify_theme(t):
            for th,ws in themes.items():
                if any(w in t for w in ws): return th
            return "Other / أخرى"

        df["Theme"]=df["__clean"].apply(classify_theme)
        df=df[df["Theme"]!="Other / أخرى"]
        counts=df["Theme"].value_counts().reset_index()
        counts.columns=["Theme","Count"]
        counts["%"]=counts["Count"]/counts["Count"].sum()*100
        counts["Cum%"]=counts["%"].cumsum()
        counts["Color"]=np.where(counts["Cum%"]<=80,"#e74c3c","#95a5a6")

        st.dataframe(counts.style.format({"%":"{:.1f}","Cum%":"{:.1f}"}))
        fig=go.Figure()
        fig.add_bar(x=counts["Theme"],y=counts["Count"],marker_color=counts["Color"],name="Count")
        fig.add_scatter(x=counts["Theme"],y=counts["Cum%"],name="Cumulative %",yaxis="y2",mode="lines+markers")
        fig.update_layout(title="Pareto — المحاور الرئيسية",yaxis=dict(title="عدد الملاحظات"),
                          yaxis2=dict(title="النسبة التراكمية (%)",overlaying="y",side="right"),bargap=0.2)
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.warning("⚠️ لا يوجد عمود نصي لتحليل Pareto.")
