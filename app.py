#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard — v8.2 (Executive Edition)
Unified | Secure | Multi-Center | Lookup | KPI Gauges | Pareto | Services Overview
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, re, os
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
    "Executive Council": {"password": "admin2025", "role": "admin", "file": None},  # الأمانة العامة
}

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="لوحة مؤشرات رضا المتعاملين — الإصدار 8.2", layout="wide")
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

if role == "admin":
    st.markdown("### 🏛️ وضع الأمانة العامة")
    target_center = st.selectbox("اختر المركز:", ["All Centers (Master)"] + [c for c in USER_KEYS if c != "Executive Council"])
    file_path = "Centers_Master.csv" if target_center == "All Centers (Master)" else USER_KEYS[target_center]["file"]
else:
    file_path = USER_KEYS[center]["file"]
    st.markdown(f"### 📊 لوحة مركز {center}")

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
# FUNCTIONS
# =========================================================
def series_to_percent(vals):
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
    promoters = (s>=9).sum(); detractors = (s<=6).sum()
    return (promoters - detractors)/len(s)*100

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
tab_data, tab_sample, tab_kpis, tab_services, tab_compare, tab_pareto = st.tabs([
    "📁 البيانات","📈 توزيع العينة","📊 المؤشرات","📋 الخدمات","🏛️ مقارنة المراكز","💬 Pareto"
])

# =========================================================
# 📁 DATA TAB
# =========================================================
with tab_data:
    st.subheader("📁 البيانات بعد الفلاتر الحالية")
    st.dataframe(df, use_container_width=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Filtered_Data")
    st.download_button("📥 تنزيل البيانات الحالية (Excel)", data=buffer.getvalue(),
                       file_name=f"Filtered_Data_{ts}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
        title = f"{col.replace('_name','')} — {total:,} رد"
        if chart_type == "دائري Pie":
            fig = px.pie(counts, names=col, values="Count", hole=0.3, title=title, color_discrete_sequence=PASTEL)
            fig.update_traces(text=counts["Count"], textinfo="value+label")
        else:
            fig = px.bar(counts, x=col, y="Count", text="Count", title=title, color=col, color_discrete_sequence=PASTEL)
            fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📊 KPIs TAB
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
# 📋 SERVICES TAB (ADMIN OR CENTER) — PLOTLY TABLE VERSION
# =========================================================
with tab_services:
    st.subheader("📋 تحليل الخدمات")

    def plot_service_table(df_in):
        """عرض جدول الخدمات بألوان تفاعلية حسب CSAT"""
        df_plot = df_in.copy()
        df_plot["CSAT_color"] = np.where(df_plot["CSAT"] >= 80, "#c8f7c5",
                                 np.where(df_plot["CSAT"] >= 60, "#fff3b0", "#f5b7b1"))
        header_color = "#2c3e50"
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df_plot.columns),
                        fill_color=header_color,
                        align='center', font=dict(color='white', size=13)),
            cells=dict(values=[df_plot[c] for c in df_plot.columns],
                       fill_color=[[c for c in df_plot["CSAT_color"]] for _ in df_plot.columns],
                       align='center', font=dict(size=12)))
        ])
        fig.update_layout(height=400, margin=dict(l=5, r=5, t=30, b=5))
        st.plotly_chart(fig, use_container_width=True)

    if role == "admin":
        combined = []
        for c, info in USER_KEYS.items():
            if info["file"] and Path(info["file"]).exists():
                df_c = pd.read_csv(info["file"], encoding="utf-8", low_memory=False)
                df_c["Center"] = c
                combined.append(df_c)
        if combined:
            all_df = pd.concat(combined, ignore_index=True)
            if "SERVICE_name" in all_df.columns:
                service_summary = all_df.groupby(["Center", "SERVICE_name"]).agg({
                    "Dim6.1": series_to_percent,
                    "Dim6.2": series_to_percent
                }).reset_index().rename(columns={"Dim6.1": "CSAT", "Dim6.2": "CES"})
                service_summary = service_summary.sort_values(by="CSAT", ascending=False)
                plot_service_table(service_summary)

                fig = px.bar(service_summary.sort_values("CSAT", ascending=False),
                             x="SERVICE_name", y="CSAT", color="Center",
                             title="الخدمات مرتبة حسب مؤشر CSAT",
                             color_discrete_sequence=PASTEL)
                fig.update_layout(xaxis_title="الخدمة", yaxis_title="CSAT (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ لم يتم العثور على عمود SERVICE_name في الملفات.")
        else:
            st.warning("⚠️ لا توجد ملفات مراكز صالحة للعرض.")
    else:
        if "SERVICE_name" in df.columns:
            service_summary = df.groupby("SERVICE_name").agg({
                "Dim6.1": series_to_percent,
                "Dim6.2": series_to_percent
            }).reset_index().rename(columns={"Dim6.1": "CSAT", "Dim6.2": "CES"})
            service_summary = service_summary.sort_values(by="CSAT", ascending=False)
            plot_service_table(service_summary)
        else:
            st.warning("⚠️ لا توجد بيانات خدمات في هذا المركز.")



# =========================================================
# 🏛️ CENTER COMPARISON TAB
# =========================================================
with tab_compare:
    st.subheader("🏛️ مقارنة المراكز")

    try:
        df_master = pd.read_csv("Centers_Master.csv", encoding="utf-8")

        # 🔍 اكتشاف الأعمدة تلقائيًا (حتى لو تغيرت الأسماء)
        col_map = {}
        for c in df_master.columns:
            c_low = c.lower().strip()
            if "center" in c_low: col_map[c] = "Center"
            elif "csat" in c_low or "dim6.1" in c_low: col_map[c] = "CSAT"
            elif "ces" in c_low or "dim6.2" in c_low: col_map[c] = "CES"
            elif "nps" in c_low or "recommend" in c_low: col_map[c] = "NPS"
        df_master.rename(columns=col_map, inplace=True)

        # ✅ اختيار الأعمدة الموجودة فعليًا
        expected_cols = ["Center", "CSAT", "CES", "NPS"]
        existing = [c for c in expected_cols if c in df_master.columns]
        if existing:
            df_master = df_master[existing].sort_values(by=existing[1], ascending=False)
            st.dataframe(df_master, use_container_width=True)

            fig = px.bar(df_master, x="Center", y="CSAT", color="CSAT",
                         color_continuous_scale=["#f5b7b1", "#fcf3cf", "#c8f7c5"],
                         title="ترتيب المراكز حسب CSAT")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"⚠️ لم يتم العثور على الأعمدة المطلوبة في الملف. الأعمدة المتاحة: {', '.join(df_master.columns)}")

    except Exception as e:
        st.warning(f"⚠️ تعذر تحميل ملف المقارنة: {e}")



# =========================================================
# 💬 PARETO TAB
# =========================================================
with tab_pareto:
    st.subheader("💬 تحليل نصوص الملاحظات (Pareto المحاور الفعلية)")
    text_cols=[c for c in df.columns if any(k in c.lower() for k in ["most_unsat","comment","ملاحظ","reason"])]
    if text_cols:
        col=text_cols[0]
        df["__clean"]=df[col].astype(str).str.lower()
        df["__clean"]=df["__clean"].replace(r"[^\u0600-\u06FFA-Za-z0-9\s]"," ",regex=True)
        df["__clean"]=df["__clean"].replace(r"\s+"," ",regex=True).str.strip()

        empty_terms={""," ","لا يوجد","لايوجد","لا شيء","no","none","nothing","جيد","ممتاز","ok"}
        df=df[~df["__clean"].isin(empty_terms)]
        df=df[df["__clean"].apply(lambda x: len(x.split())>=3)]

        themes={
            "Parking / مواقف السيارات":["موقف","مواقف","parking","السيارات"],
            "Waiting / الانتظار":["انتظار","بطء","delay","slow"],
            "Staff / الموظفون":["موظف","تعامل","staff"],
            "Fees / الرسوم":["رسوم","دفع","fee"],
            "Process / الإجراءات":["اجراء","process","انجاز"],
            "Platform / المنصة":["تطبيق","app","system","website"],
            "Facility / المكان":["مكان","نظافة","ازدحام"],
            "Communication / التواصل":["رد","تواصل","اتصال"]
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



