#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard — v10.5
Unified | Bilingual | Lookup Merge | KPIs | Pareto | Services Overview
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import re, io

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
st.set_page_config(page_title="لوحة تجربة المتعاملين — رأس الخيمة", layout="wide")
PASTEL = px.colors.qualitative.Pastel

# =========================================================
# LANGUAGE SELECTION
# =========================================================
lang = st.sidebar.radio("🌍 اللغة / Language", ["العربية", "English"], index=0)
if lang == "العربية":
    st.markdown("""
        <style>
        html, body, [class*="css"] {direction:rtl;text-align:right;font-family:"Tajawal","Cairo","Segoe UI";}
        </style>
    """, unsafe_allow_html=True)

# =========================================================
# LOGIN SECTION
# =========================================================
st.sidebar.header("🏢 اختر المركز / Select Center")
selected_center = st.sidebar.selectbox("Select Center / اختر المركز", list(USER_KEYS.keys()))

if "authorized" not in st.session_state:
    st.session_state.update({"authorized": False, "center": None, "role": None, "file": None})

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

center = st.session_state.get("center")
role = st.session_state.get("role")

st.markdown(f"### 📊 لوحة مركز {center}")

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
# LOOKUP TABLES MERGE
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
    col_upper = col.strip().upper()
    if col_upper in lookup_catalog:
        tbl = lookup_catalog[col_upper]
        merge_key = "CODE" if "CODE" in tbl.columns else tbl.columns[0]
        lang_col = "ARABIC" if lang == "العربية" else "ENGLISH"
        if lang_col in tbl.columns:
            df = df.merge(tbl[[merge_key, lang_col]], how="left", left_on=col, right_on=merge_key)
            df.rename(columns={lang_col: f"{col}_name"}, inplace=True)
            df.drop(columns=[merge_key], inplace=True, errors="ignore")

# =========================================================
# COMBINE CODE + MEANING (BILINGUAL)
# =========================================================
def combine_code_and_name(df_in):
    df_out = df_in.copy()
    for col in df_out.columns:
        if col.endswith("_name"):
            base = col.replace("_name", "")
            if base in df_out.columns:
                df_out[f"{base}_display"] = df_out[base].astype(str) + " — " + df_out[col].astype(str)
    return df_out

df = combine_code_and_name(df)

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
filter_cols = [c for c in df.columns if c.endswith("_name") and df[c].nunique() > 1]
filters = {}
with st.sidebar.expander("🎛️ الفلاتر / Filters", expanded=False):
    for col in filter_cols:
        options = sorted(df[col].dropna().unique().tolist())
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
# 📁 DATA TAB (With Arabic/English Questions Headers)
# =========================================================
# =========================================================
# 📁 DATA TAB — Arabic/English Headers (Fixed)
# =========================================================
with tab_data:
    st.subheader("📁 البيانات بعد الفلاتر")

    questions_map_ar, questions_map_en = {}, {}
    if "QUESTIONS" in lookup_catalog:
        qtbl = lookup_catalog["QUESTIONS"]
        qtbl.columns = [c.upper() for c in qtbl.columns]
        if all(x in qtbl.columns for x in ["CODE", "ARABIC", "ENGLISH"]):
            questions_map_ar = dict(zip(qtbl["CODE"], qtbl["ARABIC"]))
            questions_map_en = dict(zip(qtbl["CODE"], qtbl["ENGLISH"]))

    df_display = df.copy()

    # إنشاء صفوف الوصف العربي والإنجليزي
    ar_row = [questions_map_ar.get(c, "") for c in df_display.columns]
    en_row = [questions_map_en.get(c, "") for c in df_display.columns]

    # بناء DataFrame جديد مع الصفين العلويين
    df_combined = pd.DataFrame([ar_row, en_row], columns=df_display.columns)
    df_final = pd.concat([df_combined, df_display], ignore_index=True)

    # عرض بدون تكرار رأس الجدول
    st.data_editor(
        df_final,
        use_container_width=True,
        hide_index=True,
        height=600
    )

    # حفظ ملف Excel مع نفس التنسيق
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_final.to_excel(writer, index=False, sheet_name="Filtered_Data")
    st.download_button(
        "📥 تنزيل البيانات (Excel)",
        data=buffer.getvalue(),
        file_name=f"Filtered_Data_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# =========================================================
# 📈 SAMPLE TAB
# =========================================================
with tab_sample:
    st.subheader("📈 توزيع العينة حسب الفئات المحددة")
    total = len(df)
    if total == 0:
        st.warning("⚠️ لا توجد بيانات.")
        st.stop()

    chart_option = st.selectbox("📊 اختر نوع العرض:", ["Pie","Bar","Horizontal","Grid"], index=1)

    named_cols = list(filters.keys())

    for col in named_cols:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "عدد الردود"]
        if chart_option == "Pie":
            fig = px.pie(counts, names=col, values="عدد الردود", hole=0.3, color_discrete_sequence=PASTEL)
        elif chart_option == "Bar":
            fig = px.bar(counts, x=col, y="عدد الردود", text="عدد الردود", color=col, color_discrete_sequence=PASTEL)
        elif chart_option == "Horizontal":
            fig = px.bar(counts, y=col, x="عدد الردود", orientation="h", text="عدد الردود", color=col, color_discrete_sequence=PASTEL)
        else:
            st.dataframe(counts, use_container_width=True)
            continue
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📊 KPIs TAB
# =========================================================
with tab_kpis:
    st.subheader("📊 المؤشرات الرئيسية (CSAT / CES / NPS)")
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
                   'steps': [{'range': [0, 60], 'color': '#f5b7b1'},
                             {'range': [60, 80], 'color': '#fcf3cf'},
                             {'range': [80, 100], 'color': '#c8f7c5'}],
                   'bar': {'color': '#2ecc71'}}))
        col.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📋 SERVICES TAB
# =========================================================
# =========================================================
# 📋 SERVICES TAB (Show Names, Not Codes)
# =========================================================
with tab_services:
    st.subheader("📋 تحليل الخدمات")

    # يفضل استخدام العمود الذي يحتوي الاسم المترجم
    for candidate in ["SERVICE_display", "SERVICE_name", "SERVICE"]:
        if candidate in df.columns:
            service_col = candidate
            break
    else:
        st.warning("⚠️ لم يتم العثور على عمود للخدمات.")
        st.stop()

    # عرض الخدمات بالاسم الكامل (الكود + الاسم)
    df_service = df.copy()
    if "QUESTIONS" in lookup_catalog:
        qtbl = lookup_catalog["QUESTIONS"]
        qtbl.columns = [c.upper() for c in qtbl.columns]
        qmap = dict(zip(qtbl["CODE"], qtbl["ARABIC"])) if lang == "العربية" else dict(zip(qtbl["CODE"], qtbl["ENGLISH"]))
        df_service[service_col] = df_service[service_col].replace(qmap)

    service_summary = (
        df_service.groupby(service_col)
                  .agg(CSAT=("Dim6.1", series_to_percent),
                       CES=("Dim6.2", series_to_percent),
                       عدد_الردود=(service_col, "count"))
                  .reset_index()
                  .sort_values("CSAT", ascending=False)
    )

    service_summary["التصنيف اللوني"] = np.select(
        [service_summary["CSAT"] >= 80, service_summary["CSAT"] >= 60],
        ["🟢 مرتفع", "🟡 متوسط"],
        default="🔴 منخفض"
    )

    st.dataframe(service_summary[[service_col, "عدد_الردود", "CSAT", "CES", "التصنيف اللوني"]],
                 use_container_width=True)

    fig = px.bar(service_summary, x=service_col, y="CSAT", text="عدد_الردود",
                 color="التصنيف اللوني",
                 color_discrete_map={"🟢 مرتفع":"#c8f7c5","🟡 متوسط":"#fcf3cf","🔴 منخفض":"#f5b7b1"},
                 title="رضا المتعاملين حسب الخدمة (CSAT)")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="الخدمة", yaxis_title="CSAT (%)")
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 💬 PARETO TAB
# =========================================================
with tab_pareto:
    st.subheader("💬 تحليل نصوص الملاحظات (Pareto)")
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
            "Parking / مواقف":["موقف","مواقف","parking","السيارات"],
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
        fig.update_layout(title="Pareto — المحاور الرئيسية",
                          yaxis=dict(title="عدد الملاحظات"),
                          yaxis2=dict(title="النسبة التراكمية (%)",overlaying="y",side="right"),
                          bargap=0.2)
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.warning("⚠️ لا يوجد عمود نصي لتحليل Pareto.")


