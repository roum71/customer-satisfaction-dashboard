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
import streamlit as st

USER_KEYS = {
    "Public Services Department": {
        "password": st.secrets["users"]["Public_Services_Department"],
        "role": "center",
        "file": "Center_Public_Services.csv"
    },
    "Ras Al Khaimah Municipality": {
        "password": st.secrets["users"]["Ras_Al_Khaimah_Municipality"],
        "role": "center",
        "file": "Center_RAK_Municipality.csv"
    },
    "Sheikh Saud Center-Ras Al Khaimah Courts": {
        "password": st.secrets["users"]["Sheikh_Saud_Center"],
        "role": "center",
        "file": "Center_Sheikh_Saud_Courts.csv"
    },
    "Sheikh Saqr Center-Ras Al Khaimah Courts": {
        "password": st.secrets["users"]["Sheikh_Saqr_Center"],
        "role": "center",
        "file": "Center_Sheikh_Saqr_Courts.csv"
    },
    "Executive Council": {
        "password": st.secrets["users"]["Executive_Council"],
        "role": "admin",
        "file": "Centers_Master.csv"
    }
}


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="لوحة تجربة المتعاملين — رأس الخيمة", layout="wide")
PASTEL = px.colors.qualitative.Pastel

# =========================================================
# 🏛️ HEADER — شعار الأمانة العامة + عنوان التقرير الرسمي (Full Width + GitHub Link)
# =========================================================

# 🔗 ضع هنا رابط الصورة من GitHub (raw)
logo_url = "https://raw.githubusercontent.com/roum71/rakcx2025/main/assets/logo%20gsec%20full.png"

st.markdown(f"""
    <div style="text-align:center; margin-top:-40px;">
        <img src="{logo_url}" alt="RAK Executive Council Logo" style="width:950px; max-width:95%; height:auto;">
    </div>
    <div style='text-align:center; margin-top:10px;'>
        <h1 style='font-size:46px; color:#b30000; font-weight:bold; margin-bottom:0;'>تقرير تجربة المتعاملين 2025</h1>
        <h2 style='font-size:26px; color:#333; margin-top:5px;'>Customer Experience Report 2025</h2>
        <p style='color:#555; font-size:18px; margin-top:10px;'>المجلس التنفيذي – حكومة رأس الخيمة<br>The Executive Council – Government of Ras Al Khaimah</p>
    </div>
    <hr style="margin-top:20px; margin-bottom:10px;">
""", unsafe_allow_html=True)


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
# 🌍 BILINGUAL TEXT FUNCTION
# =========================================================
def bi_text(ar_text, en_text):
    """عرض النص بالعربية أو الإنجليزية بناءً على اختيار المستخدم"""
    return ar_text if lang == "العربية" else en_text

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
# 🎛️ FILTERS — الفلاتر (تتغير اللغة تلقائيًا)
# =========================================================
filter_cols = [c for c in df.columns if any(k in c.upper() for k in ["GENDER", "SERVICE", "SECTOR", "NATIONALITY", "ACADEMIC"])]
filters = {}
df_filtered = df.copy()

with st.sidebar.expander("🎛️ الفلاتر / Filters"):
    for col in filter_cols:
        lookup_name = col.strip().upper()
        mapped = False

        # 🔍 البحث عن جدول المطابقة في ملف Data_tables.xlsx
        if lookup_name in lookup_catalog:
            tbl = lookup_catalog[lookup_name]
            tbl.columns = [c.strip().upper() for c in tbl.columns]

            # تحديد الأعمدة في جدول الـ Lookup
            ar_col = next((c for c in tbl.columns if "ARABIC" in c or "SERVICE2" in c), None)
            en_col = next((c for c in tbl.columns if "ENGLISH" in c), None)
            code_col = next((c for c in tbl.columns if "CODE" in c or lookup_name in c), None)

            # تطبيق الترجمة على القيم
            if code_col and ((lang == "العربية" and ar_col) or (lang == "English" and en_col)):
                name_col = ar_col if lang == "العربية" else en_col
                name_map = dict(zip(tbl[code_col].astype(str), tbl[name_col].astype(str)))
                df_filtered[col] = df_filtered[col].astype(str).map(name_map).fillna(df_filtered[col])
                mapped = True

        if not mapped:
            st.sidebar.warning(f"⚠️ Lookup not applied for {col}")

        # 🏷️ تسمية الفلتر بالعربية أو الإنجليزية
        if lang == "العربية":
            if "GENDER" in col.upper():
                label = "النوع"
            elif "NATIONALITY" in col.upper():
                label = "الجنسية"
            elif "ACADEMIC" in col.upper():
                label = "المستوى الأكاديمي"
            elif "SERVICE" in col.upper():
                label = "الخدمة"
            elif "SECTOR" in col.upper():
                label = "القطاع"
            else:
                label = col
        else:
            if "GENDER" in col.upper():
                label = "Gender"
            elif "NATIONALITY" in col.upper():
                label = "Nationality"
            elif "ACADEMIC" in col.upper():
                label = "Academic Level"
            elif "SERVICE" in col.upper():
                label = "Service"
            elif "SECTOR" in col.upper():
                label = "Sector"
            else:
                label = col

        # 🧩 إنشاء الفلتر
        options = df_filtered[col].dropna().unique().tolist()
        selection = st.multiselect(label, options, default=options)
        filters[col] = selection

# 🔽 تطبيق الفلاتر على البيانات
for col, values in filters.items():
    df_filtered = df_filtered[df_filtered[col].isin(values)]

df = df_filtered.copy()

# =========================================================
# 📈 TABS
# =========================================================
tab_data, tab_sample, tab_kpis, tab_dimensions, tab_services, tab_pareto = st.tabs([
    bi_text("📁 البيانات", "Data"),
    bi_text("📈 توزيع العينة", "Sample Distribution"),
    bi_text("📊 المؤشرات", "KPIs"),
    bi_text("🧩 الأبعاد", "Dimensions"),
    bi_text("📋 الخدمات", "Services"),
    bi_text("💬مزعجات", "Pain Points")
])

# =========================================================
# 📁 DATA TAB — Multi-language headers
# =========================================================
with tab_data:
    st.subheader("📁 البيانات الخام /Raw Data")

    questions_map_ar, questions_map_en = {}, {}
    if "QUESTIONS" in lookup_catalog:
        qtbl = lookup_catalog["QUESTIONS"]
        qtbl.columns = [c.strip().upper() for c in qtbl.columns]
        code_col = next((c for c in qtbl.columns if "CODE" in c or "DIMENSION" in c), None)
        ar_col = next((c for c in qtbl.columns if "ARABIC" in c or c == "ARABIC"), None)
        en_col = next((c for c in qtbl.columns if "ENGLISH" in c or c == "ENGLISH"), None)

        if code_col and ar_col and en_col:
            qtbl["CODE_NORM"] = qtbl[code_col].astype(str).str.strip().str.upper()
            questions_map_ar = dict(zip(qtbl["CODE_NORM"], qtbl[ar_col]))
            questions_map_en = dict(zip(qtbl["CODE_NORM"], qtbl[en_col]))

    df_display = df.copy()
    df_display.columns = [c.strip() for c in df_display.columns]
    ar_row = [questions_map_ar.get(c.strip().upper(), "") for c in df_display.columns]
    en_row = [questions_map_en.get(c.strip().upper(), "") for c in df_display.columns]
    df_final = pd.concat([pd.DataFrame([ar_row, en_row], columns=df_display.columns), df_display], ignore_index=True)

    st.dataframe(df_final, use_container_width=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_final.to_excel(writer, index=False)
    st.download_button("📥 تنزيل البيانات", buffer.getvalue(), file_name=f"Filtered_Data_{ts}.xlsx")

# =========================================================
# 📈 SAMPLE TAB — توزيع العينة (ثنائي اللغة مع عناوين ديناميكية)
# =========================================================
with tab_sample:
    st.subheader(bi_text("📈 توزيع العينة", "Sample Distribution"))

    # 🧮 إجمالي الردود
    total = len(df)
    st.markdown(f"### 🧮 {bi_text('إجمالي الردود:', 'Total Responses:')} {total:,}")

    # 🟩 نوع الرسم البياني
    chart_type = st.radio(
        bi_text("📊 نوع الرسم البياني", "📊 Chart Type"),
        [bi_text("مخطط دائري (Pie Chart)", "Pie Chart"),
         bi_text("مخطط أعمدة (Bar Chart)", "Bar Chart"),
         bi_text("شبكي / مصفوفة (Grid / Matrix)", "Grid / Matrix")],
        index=1,
        horizontal=True
    )

    # 🟨 طريقة العرض
    value_type = st.radio(
        bi_text("📏 طريقة العرض", "📏 Display Mode"),
        [bi_text("الأعداد (Numbers)", "Numbers"),
         bi_text("النسب المئوية (Percentages)", "Percentages")],
        index=1,
        horizontal=True
    )

    # 🧩 تنفيذ الرسم حسب الأعمدة المختارة
    for col in filter_cols:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "Count"]
        counts["Percentage"] = counts["Count"] / total * 100

        value_col = "Count" if "Numbers" in value_type else "Percentage"

        # 🏷️ اختيار التسمية بناءً على اللغة
        if lang == "العربية":
            if col.upper() == "GENDER":
                col_label = "النوع"
            elif col.upper() == "NATIONALITY":
                col_label = "الجنسية"
            elif "ACADEMIC" in col.upper():
                col_label = "المستوى الأكاديمي"
            elif "SECTOR" in col.upper():
                col_label = "القطاع"
            elif "SERVICE" in col.upper():
                col_label = "الخدمة"
            else:
                col_label = col
            st.markdown(f"### {col_label} — {total:,} ردود")
            graph_title = f"توزيع {col_label}"
            x_title = "الفئة"
            y_title = "النسبة المئوية (%)" if value_col == "Percentage" else "العدد"

        else:  # English
            if col.upper() == "GENDER":
                col_label = "Gender"
            elif col.upper() == "NATIONALITY":
                col_label = "Nationality"
            elif "ACADEMIC" in col.upper():
                col_label = "Academic Level"
            elif "SERVICE" in col.upper():
                col_label = "Service"
            else:
                col_label = col
            st.markdown(f"### {col_label} — {total:,} Responses")
            graph_title = f"Distribution of {col_label}"
            x_title = "Category"
            y_title = "Percentage (%)" if value_col == "Percentage" else "Count"

        # 🥧 Pie Chart
        if "Pie" in chart_type:
            fig = px.pie(
                counts,
                names=col,
                values=value_col,
                hole=0.3,
                title=graph_title,
                color_discrete_sequence=PASTEL
            )
            fig.update_traces(
                texttemplate="%{label}<br>%{percent:.1%}" if value_col == "Percentage" else "%{label}<br>%{value}",
                textposition="inside",
                textfont_size=14
            )
            fig.update_layout(title_x=0.5, title_font=dict(size=20))
            st.plotly_chart(fig, use_container_width=True)

        # 📊 Bar Chart
        elif "Bar" in chart_type:
            fig = px.bar(
                counts,
                x=col,
                y=value_col,
                text=value_col,
                color=col,
                color_discrete_sequence=PASTEL,
                title=graph_title
            )
            fig.update_traces(
                texttemplate="%{text:.1f}" if value_col == "Percentage" else "%{text}",
                textposition="outside"
            )
            fig.update_layout(
                xaxis_title=x_title,
                yaxis_title=y_title,
                title_x=0.5,
                title_font=dict(size=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        # 🧩 Grid / Matrix View
        else:
            st.write(f"### 🧩 {bi_text('عرض شبكي —', 'Grid View —')} {col_label}")
            matrix = counts[[col, "Count", "Percentage"]].copy()
            matrix.columns = [
                bi_text("القيمة", "Value"),
                bi_text("العدد", "Count"),
                bi_text("النسبة المئوية", "Percentage")
            ]
            st.dataframe(
                matrix.style.format({bi_text("النسبة المئوية", "Percentage"): "{:.1f}%"}),
                use_container_width=True
            )
# =========================================================
# 📊 KPIs TAB — السعادة / القيمة / صافي نقاط الترويج
# =========================================================
with tab_kpis:
    st.subheader(bi_text("📊 مؤشرات الأداء الرئيسية (السعادة / القيمة / صافي نقاط الترويج)", 
                         "Key Performance Indicators (Happiness / Value / NPS)"))
    st.info(bi_text("يعرض هذا القسم نتائج المؤشرات الثلاثة مع تدرج الألوان وفقًا لأفضل الممارسات.",
                    "This section shows the three key indicators with color bins aligned to best practices."))

    # 🧮 حساب المؤشرات من البيانات
    csat = series_to_percent(df.get("Dim6.1", pd.Series(dtype=float)))   # Happiness
    ces = series_to_percent(df.get("Dim6.2", pd.Series(dtype=float)))    # Value
    nps, prom, passv, detr = detect_nps(df)                              # NPS

    # =========================================================
    # 🎨 تدرج الألوان والأوصاف حسب اللغة
    # =========================================================
    def get_color_and_label(score, metric_type, lang="العربية"):
        if metric_type in ["CSAT", "CES"]:
            if score < 70:
                color, label = "#FF6B6B", ("ضعيف جدًا" if lang == "العربية" else "Very Poor")
            elif score < 80:
                color, label = "#FFD93D", ("بحاجة إلى تحسين" if lang == "العربية" else "Needs Improvement")
            elif score < 90:
                color, label = "#6BCB77", ("جيد" if lang == "العربية" else "Good")
            else:
                color, label = "#4D96FF", ("ممتاز" if lang == "العربية" else "Excellent")
        else:  # NPS logic
            if score < 0:
                color, label = "#FF6B6B", ("ضعيف جدًا" if lang == "العربية" else "Very Poor")
            elif score < 30:
                color, label = "#FFD93D", ("ضعيف" if lang == "العربية" else "Fair")
            elif score < 60:
                color, label = "#6BCB77", ("جيد" if lang == "العربية" else "Good")
            else:
                color, label = "#4D96FF", ("ممتاز" if lang == "العربية" else "Excellent")
        return color, label

    # =========================================================
    # 🧭 دالة إنشاء الرسم Gauge
    # =========================================================
    def create_gauge(score, metric_type, lang="العربية"):
        color, label = get_color_and_label(score, metric_type, lang)
        if metric_type in ["CSAT", "CES"]:
            title = "السعادة / Happiness" if metric_type == "CSAT" else "القيمة / Value"
            axis_range = [0, 100]
            steps = [
                {'range': [0, 70], 'color': '#FF6B6B'},
                {'range': [70, 80], 'color': '#FFD93D'},
                {'range': [80, 90], 'color': '#6BCB77'},
                {'range': [90, 100], 'color': '#4D96FF'}
            ]
        else:
            title = "صافي نقاط الترويج / NPS"
            axis_range = [-100, 100]
            steps = [
                {'range': [-100, 0], 'color': '#FF6B6B'},
                {'range': [0, 30], 'color': '#FFD93D'},
                {'range': [30, 60], 'color': '#6BCB77'},
                {'range': [60, 100], 'color': '#4D96FF'}
            ]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score if not np.isnan(score) else 0,
            number={'suffix': "٪" if metric_type != "NPS" else ""},
            title={'text': title, 'font': {'size': 18}},
            gauge={
                'axis': {'range': axis_range},
                'bar': {'color': color},
                'steps': steps
            }
        ))
        fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=30))
        return fig, label

  # =========================================================
# 📈 عرض المؤشرات الثلاثة (السعادة / القيمة / NPS)
# =========================================================
c1, c2, c3 = st.columns(3)
for col, val, mtype in zip([c1, c2, c3], [csat, ces, nps], ["CSAT", "CES", "NPS"]):
    fig, label = create_gauge(val, mtype, lang)
    col.plotly_chart(fig, use_container_width=True)

    # 🧮 تحديد اللون الخاص بالتفسير نفسه
    color, _ = get_color_and_label(val, mtype, lang)
    text_color = f"color:{color};font-weight:bold;"

    # 🔎 تفسير مخصص للـ NPS
    if mtype == "NPS":
        if lang == "العربية":
            if val < 0:
                detail = "نتيجة سلبية تشير إلى أن عدد المعارضين يفوق عدد المروجين."
            elif val < 30:
                detail = "نتيجة ضعيفة — رضا العملاء محدود وعدد المروجين منخفض."
            elif val < 60:
                detail = "نتيجة جيدة — أغلب العملاء راضون والمروجون أكثر من المعارضين."
            else:
                detail = "نتيجة ممتازة — ولاء العملاء مرتفع جدًا ومعظمهم مروجون للخدمة."
            
            col.markdown(
                f"<p style='{text_color}'>🔎 التفسير: {label}<br>{detail}<br>"
                f"المروجون: {prom:.1f}% | المحايدون: {passv:.1f}% | المعارضون: {detr:.1f}%</p>",
                unsafe_allow_html=True
            )

        else:
            if val < 0:
                detail = "Negative score — more detractors than promoters."
            elif val < 30:
                detail = "Low score — limited satisfaction and few promoters."
            elif val < 60:
                detail = "Good score — most customers are satisfied, promoters exceed detractors."
            else:
                detail = "Excellent score — strong loyalty and many promoters."
            
            col.markdown(
                f"<p style='{text_color}'>🔎 Interpretation: {label}<br>{detail}<br>"
                f"Promoters: {prom:.1f}% | Passives: {passv:.1f}% | Detractors: {detr:.1f}%</p>",
                unsafe_allow_html=True
            )

    # 🧠 تفسير لبقية المؤشرات (CSAT و CES)
    else:
        if lang == "العربية":
            text = f"🔎 التفسير: {label}"
        else:
            text = f"🔎 Interpretation: {label}"
        col.markdown(f"<p style='{text_color}'>{text}</p>", unsafe_allow_html=True)

# =========================================================
# 🧩 DIMENSIONS TAB
# =========================================================

with tab_dimensions:
    st.subheader(bi_text("🧩 تحليل الأبعاد", "Dimension Analysis"))
    st.info(bi_text("تحليل متوسط الأبعاد بناءً على استبيانات المتعاملين", 
                    "Dimension averages based on customer feedback will appear here."))

    all_dim_cols = [c for c in df.columns if re.match(r"Dim\d+\.", c.strip())]

    if not all_dim_cols:
        st.warning("⚠️ لا توجد أعمدة فرعية للأبعاد (مثل Dim1.1, Dim2.3 ...).")
    else:
        # حساب المتوسط لكل بعد رئيسي (Dim1 إلى Dim5)
        main_dims = {}
        for i in range(1, 6):
            sub_cols = [c for c in df.columns if c.startswith(f"Dim{i}.")]
            if sub_cols:
                main_dims[f"Dim{i}"] = df[sub_cols].mean(axis=1)

        for k, v in main_dims.items():
            df[k] = v

        # إعداد ملخص القيم
        summary = []
        for dim in [f"Dim{i}" for i in range(1, 6)]:
            if dim in df.columns:
                avg = series_to_percent(df[dim])
                summary.append({"Dimension": dim, "Score": avg})
        dims = pd.DataFrame(summary).dropna()

        # ربط الأسماء بالعربية أو الإنجليزية من ملف الأسئلة
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

        # ترتيب الأبعاد حسب Dim1 إلى Dim5
        order = [f"Dim{i}" for i in range(1, 6)]
        dims["Dimension"] = pd.Categorical(dims["Dimension"], categories=order, ordered=True)
        dims = dims.sort_values("Dimension")

        # تحديد اللون بناءً على النسبة
        def get_color(score):
            if score < 70:
                return "#FF6B6B"  # أحمر
            elif score < 80:
                return "#FFD93D"  # أصفر
            elif score < 90:
                return "#6BCB77"  # أخضر
            else:
                return "#4D96FF"  # أزرق

        dims["Color"] = dims["Score"].apply(get_color)

        # رسم الأعمدة حسب الترتيب واللون
        fig = px.bar(
            dims,
            x="Dimension_name" if "Dimension_name" in dims.columns else "Dimension",
            y="Score",
            text="Score",
            color="Color",
            color_discrete_map="identity",
            title="تحليل متوسط الأبعاد"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            yaxis_title="النسبة المئوية (%)",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # 🔹 إضافة وسيلة الإيضاح (الشرح)
        st.markdown("""
        **🗂️ Color Legend | وسيلة الإيضاح:**
        - 🔴 **أقل من 70٪** — منخفض / ضعيف الأداء  
        - 🟡 **من 70٪ إلى أقل من 80٪** — متوسط  
        - 🟢 **من 80٪ إلى أقل من 90٪** — جيد  
        - 🔵 **90٪ فأكثر** — ممتاز  
        """, unsafe_allow_html=True)

        # عرض الجدول
        st.dataframe(dims, use_container_width=True)

# =========================================================
# 📋 SERVICES TAB — تحليل الخدمات (Happiness / Value / NPS)
# =========================================================
with tab_services:
    st.subheader(bi_text("📋 تحليل الخدمات", "Service Analysis"))
    st.info(bi_text("مقارنة مستويات السعادة والقيمة حسب الخدمة", 
                    "Compare Happiness and Value levels per service."))

    if "SERVICE" not in df.columns:
        st.warning("⚠️ لا توجد بيانات خاصة بالخدمات.")
    else:
        df_services = df.copy()

        # 🔍 تحديد الأعمدة الخاصة بالسعادة (CSAT) والقيمة (CES) وNPS
        csat_col = next((c for c in df_services.columns if c.upper().startswith("DIM6.1")), None)
        ces_col = next((c for c in df_services.columns if c.upper().startswith("DIM6.2")), None)
        nps_col = next((c for c in df_services.columns if c.strip().upper() == "NPS"), None)

        if not csat_col or not ces_col:
            st.warning("⚠️ لم يتم العثور على الأعمدة Dim6.1 أو Dim6.2 في البيانات.")
        else:
            # 🧮 تحويل القيم من 1–5 إلى 0–100
            df_services["Happiness / سعادة (٪)"] = (df_services[csat_col] - 1) * 25
            df_services["Value / قيمة (٪)"] = (df_services[ces_col] - 1) * 25

            # 🧮 حساب NPS من العمود الموجود (0–10 مقياس)
            if nps_col:
                df_services["NPS_SCORE"] = pd.to_numeric(df_services[nps_col], errors="coerce")
                nps_summary = []
                for svc, subdf in df_services.groupby("SERVICE"):
                    valid = subdf["NPS_SCORE"].dropna()
                    if len(valid) == 0:
                        nps_summary.append((svc, np.nan))
                        continue
                    promoters = (valid >= 9).sum()
                    detractors = (valid <= 6).sum()
                    total = len(valid)
                    nps_value = ((promoters - detractors) / total) * 100
                    nps_summary.append((svc, nps_value))
                nps_df = pd.DataFrame(nps_summary, columns=["SERVICE", "NPS / صافي نقاط الترويج (٪)"])
            else:
                nps_df = pd.DataFrame(columns=["SERVICE", "NPS / صافي نقاط الترويج (٪)"])

            # 🧾 حساب المتوسط وعدد الردود لكل خدمة
            summary = (
                df_services.groupby("SERVICE")
                .agg({
                    "Happiness / سعادة (٪)": "mean",
                    "Value / قيمة (٪)": "mean",
                    csat_col: "count"
                })
                .reset_index()
                .rename(columns={csat_col: "#Responses/عدد الردود"})
            )

            # ✅ دمج نتائج NPS مع بقية المؤشرات
            summary = summary.merge(nps_df, on="SERVICE", how="left")

            # 🌐 استبدال أسماء الخدمات بالعربية / الإنجليزية من lookup
            if "SERVICE" in lookup_catalog:
                tbl = lookup_catalog["SERVICE"]
                tbl.columns = [c.strip().upper() for c in tbl.columns]
                ar_col = next((c for c in tbl.columns if "ARABIC" in c or "SERVICE2" in c), None)
                en_col = next((c for c in tbl.columns if "ENGLISH" in c), None)
                code_col = next((c for c in tbl.columns if "CODE" in c or "SERVICE" in c), None)
                if ar_col and en_col and code_col:
                    name_map = dict(zip(tbl[code_col], tbl[ar_col if lang == "العربية" else en_col]))
                    summary["SERVICE"] = summary["SERVICE"].map(name_map).fillna(summary["SERVICE"])

            # 🧹 تنسيق الأعمدة
            summary.rename(columns={"SERVICE": "الخدمة / Service"}, inplace=True)

            # 🚫 عرض فقط الخدمات التي بها 30 ردًا أو أكثر
            summary = summary[summary["#Responses/عدد الردود"] >= 30]

            # 🧭 ترتيب الجدول تنازليًا حسب السعادة
            summary = summary.sort_values("Happiness / سعادة (٪)", ascending=False)

            # ✅ تلوين الخلايا في الجدول (السعادة والقيمة فقط)
            def color_cells(val):
                try:
                    v = float(val)
                    if v < 70:
                        color = "#FF6B6B"  # أحمر
                    elif v < 80:
                        color = "#FFD93D"  # أصفر
                    elif v < 90:
                        color = "#6BCB77"  # أخضر
                    else:
                        color = "#4D96FF"  # أزرق
                    return f"background-color:{color};color:black"
                except:
                    return ""

            # 📋 عرض الجدول مع التلوين
            styled_table = (
                summary.style
                .format({
                    "Happiness / سعادة (٪)": "{:.1f}%",
                    "Value / قيمة (٪)": "{:.1f}%",
                    "NPS / صافي نقاط الترويج (٪)": "{:.1f}%",
                    "#Responses/عدد الردود": "{:,.0f}"
                })
                .applymap(color_cells, subset=["Happiness / سعادة (٪)", "Value / قيمة (٪)"])
            )
            st.dataframe(styled_table, use_container_width=True)

            # 🛈 ملاحظة توضيحية باللغتين
            st.markdown("""
            **ℹ️ ملاحظة / Note:**  
            يتم عرض الخدمات التي تحتوي على **30 ردًا أو أكثر فقط** لضمان دقة النتائج.  
            Only **services with 30 or more responses** are shown to ensure result accuracy.
            """)

            # 🎨 الرسم البياني — فقط للسعادة والقيمة
            if not summary.empty:
                df_melted = summary.melt(
                    id_vars=["الخدمة / Service", "#Responses/عدد الردود"],
                    value_vars=["Happiness / سعادة (٪)", "Value / قيمة (٪)"],
                    var_name="المؤشر",
                    value_name="القيمة"
                )

                fig = px.bar(
                    df_melted,
                    x="الخدمة / Service",
                    y="القيمة",
                    color="المؤشر",
                    barmode="group",
                    text="القيمة",
                    title=bi_text("📊 مقارنة مؤشري السعادة والقيمة حسب الخدمة", 
                                  "📊 Comparison of Happiness and Value by Service"),
                    color_discrete_sequence=PASTEL
                )

                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

                # 🎯 خط مستهدف عند 80%
                fig.add_shape(
                    type="line",
                    x0=-0.5, x1=len(summary)-0.5,
                    y0=80, y1=80,
                    line=dict(color="green", dash="dash", width=2)
                )
                fig.add_annotation(
                    xref="paper", x=1.02, y=80,
                    text=bi_text("🎯 الحد المستهدف (80%)", "🎯 Target Threshold (80%)"),
                    showarrow=False,
                    font=dict(color="green")
                )

                fig.update_layout(
                    yaxis_title=bi_text("النسبة المئوية (%)", "Percentage (%)"),
                    xaxis_title=bi_text("الخدمة / Service", "Service"),
                    legend_title=bi_text("المؤشر", "Indicator"),
                    yaxis=dict(range=[0, 100])
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(bi_text("ℹ️ لا توجد خدمات تحتوي على 30 ردًا أو أكثر.", 
                                "ℹ️ No services with 30 or more responses."))


# =========================================================
# 💬 PARETO TAB — تحليل الملاحظات (ثنائي اللغة)
# =========================================================
with tab_pareto:
    st.subheader(bi_text("💬 تحليل الملاحظات (Pareto)", "Customer Comments (Pareto)"))
    st.info(bi_text(
        "تحليل الملاحظات النوعية لتحديد أكثر الأسباب شيوعًا لعدم الرضا",
        "Qualitative analysis of comments to identify top dissatisfaction reasons."
    ))

    text_cols = [c for c in df.columns if any(k in c.lower() for k in ["comment", "ملاحظ", "unsat", "reason"])]
    if not text_cols:
        st.warning("⚠️ لا يوجد عمود نصي لتحليل Pareto.")
    else:
        col = text_cols[0]
        df["__clean"] = df[col].astype(str).str.lower()
        df["__clean"] = df["__clean"].replace(r"[^\u0600-\u06FFA-Za-z0-9\s]", " ", regex=True)
        df["__clean"] = df["__clean"].replace(r"\s+", " ", regex=True).str.strip()

        empty_terms = {"", " ", "لا يوجد", "لايوجد", "لا شيء", "no", "none", "nothing", "جيد", "ممتاز", "ok"}
        df = df[~df["__clean"].isin(empty_terms)]
        df = df[df["__clean"].apply(lambda x: len(x.split()) >= 3)]

        themes = {
            "Parking / مواقف السيارات": ["موقف", "مواقف", "parking"],
            "Waiting / الانتظار": ["انتظار", "بطء", "delay", "slow"],
            "Staff / الموظفون": ["موظف", "تعامل", "staff"],
            "Fees / الرسوم": ["رسوم", "دفع", "fee"],
            "Process / الإجراءات": ["اجراء", "process", "انجاز"],
            "Platform / المنصة": ["تطبيق", "app", "system"],
            "Facility / المكان": ["مكان", "نظافة", "ازدحام"],
            "Communication / التواصل": ["رد", "تواصل", "اتصال"]
        }

        def classify_theme(t):
            for th, ws in themes.items():
                if any(w in t for w in ws):
                    return th
            return "Other / أخرى"

        df["Theme"] = df["__clean"].apply(classify_theme)
        df = df[df["Theme"] != "Other / أخرى"]

        counts = df["Theme"].value_counts().reset_index()
        counts.columns = ["Theme", "Count"]
        counts["%"] = counts["Count"] / counts["Count"].sum() * 100
        counts["Cum%"] = counts["%"].cumsum()
        counts["Color"] = np.where(counts["Cum%"] <= 80, "#e74c3c", "#95a5a6")

        all_answers = df.groupby("Theme")["__clean"].apply(lambda x: " / ".join(x.astype(str))).reset_index()
        counts = counts.merge(all_answers, on="Theme", how="left")
        counts.rename(columns={"__clean": bi_text("جميع الإجابات", "All Responses")}, inplace=True)

        # 📋 إعداد أسماء الأعمدة حسب اللغة
        if lang == "العربية":
            counts_display = counts.rename(columns={
                "Theme": "المحور / Theme",
                "Count": "عدد الملاحظات / Count",
                "%": "النسبة %",
                "Cum%": "النسبة التراكمية %",
                bi_text("جميع الإجابات", "All Responses"): "جميع الإجابات / All Responses"
            })
        else:
            counts_display = counts.rename(columns={
                "Theme": "Theme",
                "Count": "Count",
                "%": "%",
                "Cum%": "Cum %",
                bi_text("جميع الإجابات", "All Responses"): "All Responses"
            })

        # 📊 عرض الجدول
        st.dataframe(
            counts_display.style.format({
                "النسبة %": "{:.1f}",
                "النسبة التراكمية %": "{:.1f}",
                "%": "{:.1f}",
                "Cum %": "{:.1f}"
            }),
            use_container_width=True
        )

        # 📈 رسم Pareto ثنائي اللغة
        fig = go.Figure()
        fig.add_bar(
            x=counts["Theme"],
            y=counts["Count"],
            marker_color=counts["Color"],
            name=bi_text("عدد الملاحظات", "Number of Comments")
        )
        fig.add_scatter(
            x=counts["Theme"],
            y=counts["Cum%"],
            name=bi_text("النسبة التراكمية", "Cumulative %"),
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#2e86de", width=2)
        )

        fig.update_layout(
            title=bi_text("تحليل Pareto — المحاور الرئيسية", "Pareto Analysis — Main Themes"),
            yaxis=dict(title=bi_text("عدد الملاحظات", "Number of Comments")),
            yaxis2=dict(
                title=bi_text("النسبة التراكمية (%)", "Cumulative Percentage (%)"),
                overlaying="y",
                side="right"
            ),
            xaxis=dict(
                title=bi_text("المحاور / Themes", "Themes"),
                tickfont=dict(size=12)
            ),
            bargap=0.25,
            height=600,
            title_x=0.5,
            legend_title_text=bi_text("المؤشر", "Indicator")
        )

        st.plotly_chart(fig, use_container_width=True)

        # 🧭 وسيلة الإيضاح (Legend)
        legend_html = """
        <div style='background-color:#f9f9f9; border:1px solid #ddd; border-radius:8px; padding:10px; margin-top:10px;'>
          <p style='font-size:15px; margin:0;'>
            <b>🎨 وسيلة الإيضاح / Legend:</b><br>
            🔴 <b>الأحمر:</b> الأسباب الأكثر تأثيرًا وتشكل معًا <b>حتى 80٪</b> من الملاحظات.<br>
            ⚪ <b>الرمادي:</b> الأسباب الأقل تأثيرًا وتشكل النسبة المتبقية من الملاحظات.
          </p>
        </div>
        """ if lang == "العربية" else """
        <div style='background-color:#f9f9f9; border:1px solid #ddd; border-radius:8px; padding:10px; margin-top:10px;'>
          <p style='font-size:15px; margin:0;'>
            <b>🎨 Legend:</b><br>
            🔴 <b>Red:</b> Most frequent causes — top <b>80%</b> of comments.<br>
            ⚪ <b>Gray:</b> Less frequent causes — remaining share of comments.
          </p>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

        # 📥 زر تنزيل ملف Excel
        pareto_buffer = io.BytesIO()
        with pd.ExcelWriter(pareto_buffer, engine="openpyxl") as writer:
            counts.to_excel(writer, index=False, sheet_name="Pareto_Results")
        st.download_button(
            "📥 تنزيل جدول Pareto (Excel)",
            data=pareto_buffer.getvalue(),
            file_name=f"Pareto_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )















