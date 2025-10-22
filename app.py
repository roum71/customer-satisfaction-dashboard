#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard — v9.0 (Unified Centers with Filters)
✅ لا يوجد All Centers
✅ تضم الفلاتر الكاملة
✅ تعمل لكل مركز على حدة (منسق أو أمانة عامة)
✅ تبويبات: البيانات / توزيع العينة / المؤشرات / الخدمات / Pareto
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
    "Public Services Department": {"password": "psd2025", "file": "Center_Public_Services.csv"},
    "Ras Al Khaimah Municipality": {"password": "rakm2025", "file": "Center_RAK_Municipality.csv"},
    "Sheikh Saud Center-Ras Al Khaimah Courts": {"password": "ssc2025", "file": "Center_Sheikh_Saud_Courts.csv"},
    "Sheikh Saqr Center-Ras Al Khaimah Courts": {"password": "ssq2025", "file": "Center_Sheikh_Saqr_Courts.csv"},
    "Executive Council": {"password": "admin2025", "file": None},
}

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="لوحة تجربة المتعاملين — رأس الخيمة", layout="wide")
PASTEL = px.colors.qualitative.Pastel

# =========================================================
# اللغة
# =========================================================
lang = st.sidebar.radio("🌍 اللغة / Language", ["العربية", "English"], index=0)
if lang == "العربية":
    st.markdown("""
        <style>
        html, body, [class*="css"] {direction:rtl;text-align:right;font-family:"Tajawal","Cairo","Segoe UI";}
        </style>
    """, unsafe_allow_html=True)

# =========================================================
# تسجيل الدخول
# =========================================================
st.sidebar.header("🔐 تسجيل الدخول")
center_options = [c for c in USER_KEYS if c != "Executive Council"]

selected_user = st.sidebar.selectbox("👤 اختر المستخدم", list(USER_KEYS.keys()))
password = st.sidebar.text_input("كلمة المرور / Password", type="password")

if "authorized" not in st.session_state:
    st.session_state.update({"authorized": False, "center": None, "file": None})

if not st.session_state["authorized"] or st.session_state["center"] != selected_user:
    if password == USER_KEYS[selected_user]["password"]:
        st.session_state["authorized"] = True
        st.session_state["center"] = selected_user
        st.success("✅ تم تسجيل الدخول بنجاح.")
        st.rerun()
    elif password:
        st.error("🚫 كلمة المرور غير صحيحة.")
        st.stop()
    else:
        st.warning("يرجى إدخال كلمة المرور.")
        st.stop()

center = st.session_state["center"]

# =========================================================
# اختيار المركز (للأمانة العامة فقط)
# =========================================================
if center == "Executive Council":
    st.markdown("### 🏛️ الأمانة العامة — اختر مركزًا لعرض بياناته")
    target_center = st.selectbox("اختر المركز:", center_options)
    file_path = USER_KEYS[target_center]["file"]
else:
    target_center = center
    file_path = USER_KEYS[center]["file"]

if not file_path or not Path(file_path).exists():
    st.error("❌ ملف المركز غير موجود.")
    st.stop()

# =========================================================
# تحميل البيانات
# =========================================================
@st.cache_data
def safe_read(file):
    try:
        return pd.read_csv(file, encoding="utf-8", low_memory=False)
    except Exception as e:
        st.error(f"خطأ في تحميل الملف: {e}")
        return pd.DataFrame()

df = safe_read(file_path)
if df.empty:
    st.warning("⚠️ لا توجد بيانات في هذا الملف.")
    st.stop()

st.markdown(f"### 📊 لوحة مركز: **{target_center}**")

# =========================================================
# دوال المساعدة
# =========================================================
def series_to_percent(vals):
    s = pd.to_numeric(vals, errors="coerce").dropna()
    if s.empty: return np.nan
    mx = s.max()
    if mx <= 5: return ((s - 1) / 4 * 100).mean()
    elif mx <= 10: return ((s - 1) / 9 * 100).mean()
    else: return s.mean()

def detect_nps(df_in):
    cands = [c for c in df_in.columns if ("nps" in c.lower()) or ("recommend" in c.lower())]
    if not cands: return np.nan
    s = pd.to_numeric(df_in[cands[0]], errors="coerce").dropna()
    if s.empty: return np.nan
    promoters = (s >= 9).sum(); detractors = (s <= 6).sum()
    return (promoters - detractors) / len(s) * 100

# =========================================================
# الفلاتر
# =========================================================
filter_cols = [c for c in df.columns if c.endswith("_name") or c.lower() in ["gender","service","sector","nationality"]]
filters = {}
with st.sidebar.expander("🎛️ الفلاتر / Filters", expanded=False):
    for col in filter_cols:
        options = sorted(df[col].dropna().unique().tolist())
        if len(options) > 1:
            selected = st.multiselect(f"{col}", options, default=options)
            filters[col] = selected
for col, vals in filters.items():
    df = df[df[col].isin(vals)]

# =========================================================
# التبويبات
# =========================================================
tab_data, tab_sample, tab_kpis, tab_services, tab_pareto = st.tabs([
    "📁 البيانات","📈 توزيع العينة","📊 المؤشرات","📋 الخدمات","💬 Pareto"
])

# =========================================================
# 📁 البيانات
# =========================================================
with tab_data:
    st.subheader("📁 البيانات بعد تطبيق الفلاتر")
    st.dataframe(df, use_container_width=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Filtered")
    st.download_button("📥 تنزيل البيانات الحالية (Excel)", data=buffer.getvalue(),
                       file_name=f"{target_center}_Filtered_{ts}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =========================================================
# 📈 توزيع العينة
# =========================================================

# =========================================================
# 📈 توزيع العينة (خيارات متعددة - أعداد فقط)
# =========================================================
with tab_sample:
    st.subheader("📈 توزيع العينة حسب الفئات المحددة")
    total = len(df)
    if total == 0:
        st.warning("⚠️ لا توجد بيانات بعد الفلاتر.")
        st.stop()

    chart_option = st.selectbox(
        "📊 اختر نوع العرض:",
        ["مخطط دائري Pie","أعمدة عمودية Bar","أعمدة أفقية Horizontal Bar","أعمدة مكدسة Stacked Bar",
         "مخطط تراكمي Sunburst","خريطة حرارية Heatmap","جدول شبكي Grid"],
        index=1
    )

    # فقط الأعمدة المسماة
    named_cols = list(filters.keys())

    if chart_option == "مخطط دائري Pie":
        for col in named_cols:
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "عدد الردود"]
            fig = px.pie(counts, names=col, values="عدد الردود", hole=0.3,
                         title=f"توزيع {col}", color_discrete_sequence=PASTEL)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_option == "أعمدة عمودية Bar":
        for col in named_cols:
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "عدد الردود"]
            fig = px.bar(counts, x=col, y="عدد الردود", text="عدد الردود",
                         color=col, color_discrete_sequence=PASTEL,
                         title=f"توزيع {col}")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_option == "أعمدة أفقية Horizontal Bar":
        for col in named_cols:
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "عدد الردود"]
            fig = px.bar(counts, y=col, x="عدد الردود", text="عدد الردود",
                         orientation="h", color=col, color_discrete_sequence=PASTEL,
                         title=f"توزيع {col} (أفقي)")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_option == "جدول شبكي Grid":
        st.write("🧮 توزيع الردود حسب كل فئة:")
        for col in named_cols:
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "عدد الردود"]
            st.markdown(f"#### 📊 {col}")
            st.dataframe(counts, use_container_width=True)

# =========================================================
# 📊 المؤشرات
# =========================================================
with tab_kpis:
    st.subheader("📊 المؤشرات الرئيسية (CSAT / CES / NPS)")
    csat = series_to_percent(df.get("Dim6.1", pd.Series(dtype=float)))
    ces = series_to_percent(df.get("Dim6.2", pd.Series(dtype=float)))
    nps = detect_nps(df)
    c1, c2, c3 = st.columns(3)
    for c, val, name in zip([c1, c2, c3], [csat, ces, nps], ["CSAT", "CES", "NPS"]):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val if not np.isnan(val) else 0,
            title={'text': name},
            gauge={'axis': {'range': [0, 100]},
                   'steps': [{'range': [0, 60], 'color': '#f5b7b1'},
                             {'range': [60, 80], 'color': '#fcf3cf'},
                             {'range': [80, 100], 'color': '#c8f7c5'}],
                   'bar': {'color': '#2ecc71'}}))
        c.plotly_chart(fig, use_container_width=True)


# =========================================================
# 📋 الخدمات
# =========================================================



# =========================================================
# 📋 تحليل الخدمات — دعم جميع أسماء الأعمدة الممكنة
# =========================================================
with tab_services:
    st.subheader("📋 تحليل الخدمات")

    # 🔍 اكتشاف ذكي لعمود الخدمة
    service_candidates = [c for c in df.columns if re.search(r'(serv|خدم)', c, re.IGNORECASE)]
    if not service_candidates:
        st.warning("⚠️ لم يتم العثور على عمود يمثل اسم الخدمة. يرجى التأكد أن ملف البيانات يحتوي على عمود باسم مثل 'Service' أو 'اسم الخدمة'.")
        st.info(f"🧩 الأعمدة المتوفرة حاليًا: {', '.join(df.columns[:20])} ...")
        st.stop()

    # استخدام أول عمود مرشح
    service_col = service_candidates[0]

    # 🔹 محاولة العثور على عمود الاسم المكافئ (name أو Arabic)
    name_candidates = [c for c in df.columns if re.search(r'(name|arabic|english|اسم)', c, re.IGNORECASE) and re.search(r'(serv|خدم)', c, re.IGNORECASE)]
    service_display_col = name_candidates[0] if name_candidates else service_col

    # 🔹 حساب مؤشرات الخدمات
    service_summary = (
        df.groupby(service_display_col)
          .agg(
              CSAT=("Dim6.1", series_to_percent),
              CES=("Dim6.2", series_to_percent),
              عدد_الردود=(service_display_col, "count")
          )
          .reset_index()
          .sort_values("CSAT", ascending=False)
    )

    # 🔹 إضافة تصنيف لوني واضح
    service_summary["التصنيف اللوني"] = np.select(
        [
            service_summary["CSAT"] >= 80,
            service_summary["CSAT"] >= 60
        ],
        ["🟢 مرتفع", "🟡 متوسط"],
        default="🔴 منخفض"
    )

    # 🔹 عرض الجدول بوضوح
    st.dataframe(
        service_summary[["التصنيف اللوني", service_display_col, "عدد_الردود", "CSAT", "CES"]]
        .style.format({"CSAT": "{:.1f}", "CES": "{:.1f}"}),
        use_container_width=True
    )

    # 🔹 رسم بياني مرتب حسب CSAT
    fig_bar = px.bar(
        service_summary.sort_values("CSAT", ascending=False),
        x=service_display_col, y="CSAT",
        text="عدد_الردود",
        color="التصنيف اللوني",
        color_discrete_map={"🟢 مرتفع": "#c8f7c5", "🟡 متوسط": "#fcf3cf", "🔴 منخفض": "#f5b7b1"},
        title="📊 مؤشر رضا المتعاملين (CSAT) حسب الخدمة"
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(xaxis_title="الخدمة", yaxis_title="CSAT (%)")
    st.plotly_chart(fig_bar, use_container_width=True)

# =========================================================
# 💬 Pareto
# =========================================================
with tab_pareto:
    st.subheader("💬 تحليل نصوص الملاحظات (Pareto)")
    text_cols = [c for c in df.columns if any(k in c.lower() for k in ["unsat","comment","reason","ملاحظ"])]
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




