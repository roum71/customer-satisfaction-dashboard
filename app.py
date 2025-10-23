#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard — v10.7 (Fixed & Stable)
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

df_filtered = df.copy()

with st.sidebar.expander("🎛️ الفلاتر / Filters"):
    for col in filter_cols:
        lookup_name = col.strip().upper()
        mapped = False
        if lookup_name in lookup_catalog:
            tbl = lookup_catalog[lookup_name]
            tbl.columns = [c.strip().upper() for c in tbl.columns]
            
            # Detect columns (case-insensitive)
            ar_col = next((c for c in tbl.columns if "ARABIC" in c or "SERVICE2" in c), None)
            en_col = next((c for c in tbl.columns if "ENGLISH" in c), None)
            code_col = next((c for c in tbl.columns if "CODE" in c or lookup_name in c), None)

            if code_col and ((lang == "العربية" and ar_col) or (lang == "English" and en_col)):
                name_col = ar_col if lang == "العربية" else en_col
                name_map = dict(zip(tbl[code_col].astype(str), tbl[name_col].astype(str)))
                df_filtered[col] = df_filtered[col].astype(str).map(name_map).fillna(df_filtered[col])
                mapped = True
        
        if not mapped:
            st.sidebar.warning(f"⚠️ Lookup not applied for {col}")

        options = df_filtered[col].dropna().unique().tolist()
        selection = st.multiselect(col, options, default=options)
        filters[col] = selection

for col, values in filters.items():
    df_filtered = df_filtered[df_filtered[col].isin(values)]

df = df_filtered.copy()


# =========================================================
# 📈 TABS
# =========================================================
tab_data, tab_sample, tab_kpis, tab_dimensions, tab_services, tab_pareto = st.tabs(
    ["📁 البيانات", "📈 توزيع العينة", "📊 المؤشرات", "🧩 الأبعاد", "📋 الخدمات", "💬 Pareto"]
)

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
        counts["%"] = counts["Count"] / total * 100
        title = f"{col} — {total:,} رد"
        if chart_type == "دائري Pie":
            fig = px.pie(counts, names=col, values="Count", hole=0.3, title=title, color_discrete_sequence=PASTEL)
        else:
            fig = px.bar(counts, x=col, y="Count", text="Count", color=col, color_discrete_sequence=PASTEL)
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 📊 KPIs TAB — 3 gauges + NPS breakdown
# =========================================================
with tab_kpis:
    st.subheader("📊 المؤشرات الرئيسية (CSAT / CES / NPS)")
    csat = series_to_percent(df.get("Dim6.1", pd.Series(dtype=float)))
    ces = series_to_percent(df.get("Dim6.2", pd.Series(dtype=float)))
    nps, prom, passv, detr = detect_nps(df)

    c1, c2, c3 = st.columns(3)
    for col, val, name in zip([c1, c2, c3], [csat, ces, nps], ["CSAT", "CES", "NPS"]):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val if not np.isnan(val) else 0,
            title={'text': name},
            gauge={'axis': {'range': [0, 100]},
                   'steps': [{'range': [0, 60], 'color': '#f5b7b1'},
                             {'range': [60, 80], 'color': '#fcf3cf'},
                             {'range': [80, 100], 'color': '#c8f7c5'}],
                   'bar': {'color': '#2ecc71'}}))
        col.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    #### 🔎 تفاصيل مؤشر NPS
    - **Promoters (المروجون):** {prom:.1f}%
    - **Passives (المحايدون):** {passv:.1f}%
    - **Detractors (المعارضون):** {detr:.1f}%
    """)

# =========================================================
# 🧩 DIMENSIONS TAB
# =========================================================
with tab_dimensions:
    st.subheader("🧩 تحليل الأبعاد")
    dim_cols = [c for c in df.columns if re.match(r"Dim\d", c)]
    if not dim_cols:
        st.warning("⚠️ لا توجد أعمدة للأبعاد (Dim).")
    else:
        summary = []
        for col in dim_cols:
            avg = series_to_percent(df[col])
            summary.append({"Dimension": col, "Score": avg})
        dims = pd.DataFrame(summary).dropna()

        if "QUESTIONS" in lookup_catalog:
            qtbl = lookup_catalog["QUESTIONS"]
            qtbl.columns = [c.strip().upper() for c in qtbl.columns]
            code_col = next((c for c in qtbl.columns if "CODE" in c or "DIMENSION" in c), None)
            ar_col = next((c for c in qtbl.columns if "ARABIC" in c), None)
            en_col = next((c for c in qtbl.columns if "ENGLISH" in c), None)
            if code_col and ar_col and en_col:
                qtbl["CODE_NORM"] = qtbl[code_col].astype(str).str.strip()
                name_map = dict(zip(qtbl["CODE_NORM"],
                                    qtbl[ar_col if lang == "العربية" else en_col]))
                dims["Dimension_name"] = dims["Dimension"].map(name_map)

        fig = px.bar(dims.sort_values("Score", ascending=False),
                     x="Dimension_name" if "Dimension_name" in dims.columns else "Dimension",
                     y="Score", text="Score",
                     color_discrete_sequence=PASTEL,
                     title="تحليل متوسط الأبعاد")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(yaxis_title="النسبة المئوية (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dims, use_container_width=True)

# =========================================================
# 📋 SERVICES TAB
# =========================================================
with tab_services:
    st.subheader("📋 تحليل الخدمات")
    if "SERVICE" not in df.columns:
        st.warning("⚠️ لا يوجد عمود للخدمات.")
    else:
        svc_summary = df.groupby("SERVICE").agg({"Dim6.1":"mean","Dim6.2":"mean"}).reset_index()
        svc_summary.rename(columns={"Dim6.1":"CSAT","Dim6.2":"CES"}, inplace=True)
        st.dataframe(svc_summary, use_container_width=True)
        fig = px.bar(svc_summary, x="SERVICE", y=["CSAT","CES"], barmode="group", color_discrete_sequence=PASTEL)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 💬 PARETO TAB
# =========================================================
with tab_pareto:
    st.subheader("💬 تحليل الملاحظات (Pareto)")
    text_cols = [c for c in df.columns if any(k in c.lower() for k in ["comment","ملاحظ","unsat","reason"])]
    if not text_cols:
        st.warning("⚠️ لا يوجد عمود نصي لتحليل Pareto.")
    else:
        col = text_cols[0]
        df["__clean"] = df[col].astype(str).str.lower()
        df["__clean"] = df["__clean"].replace(r"[^\u0600-\u06FFA-Za-z0-9\s]", " ", regex=True)
        df["__clean"] = df["__clean"].replace(r"\s+", " ", regex=True).str.strip()
        empty_terms = {""," ","لا يوجد","لايوجد","لا شيء","no","none","nothing","جيد","ممتاز","ok"}
        df = df[~df["__clean"].isin(empty_terms)]
        df = df[df["__clean"].apply(lambda x: len(x.split()) >= 3)]

        themes = {
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
            for th, ws in themes.items():
                if any(w in t for w in ws):
                    return th
            return "Other / أخرى"

        df["Theme"] = df["__clean"].apply(classify_theme)
        df = df[df["Theme"] != "Other / أخرى"]

        counts = df["Theme"].value_counts().reset_index()
        counts.columns = ["Theme","Count"]
        counts["%"] = counts["Count"]/counts["Count"].sum()*100
        counts["Cum%"] = counts["%"].cumsum()
        counts["Color"] = np.where(counts["Cum%"] <= 80,"#e74c3c","#95a5a6")

        all_answers = df.groupby("Theme")["__clean"].apply(lambda x:" / ".join(x.astype(str))).reset_index()
        counts = counts.merge(all_answers,on="Theme",how="left")
        counts.rename(columns={"__clean":"جميع الإجابات"},inplace=True)

        st.dataframe(counts[["Theme","Count","%","Cum%","جميع الإجابات"]]
                     .style.format({"%":"{:.1f}","Cum%":"{:.1f}"}), use_container_width=True)

        fig = go.Figure()
        fig.add_bar(x=counts["Theme"], y=counts["Count"], marker_color=counts["Color"], name="عدد الملاحظات")
        fig.add_scatter(x=counts["Theme"], y=counts["Cum%"], name="النسبة التراكمية", yaxis="y2", mode="lines+markers")
        fig.update_layout(title="Pareto — المحاور الرئيسية",
                          yaxis=dict(title="عدد الملاحظات"),
                          yaxis2=dict(title="النسبة التراكمية (%)", overlaying="y", side="right"),
                          bargap=0.25, height=600)
        st.plotly_chart(fig, use_container_width=True)

        pareto_buffer = io.BytesIO()
        with pd.ExcelWriter(pareto_buffer, engine="openpyxl") as writer:
            counts.to_excel(writer, index=False, sheet_name="Pareto_Results")
        st.download_button("📥 تنزيل جدول Pareto (Excel)",
                           data=pareto_buffer.getvalue(),
                           file_name=f"Pareto_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")







