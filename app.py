import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

# --------------------------------------------------
# 1. PAGE CONFIG & MODERN SAAS CSS
# --------------------------------------------------
st.set_page_config(page_title="ChurnOps BI", layout="wide", initial_sidebar_state="expanded")

custom_css = """
<style>
    /* Global Theme */
    :root {
        --primary: #7C3AED;
        --secondary: #A78BFA;
        --accent: #C4B5FD;
        --bg: #0B0B12;
        --card-bg: #111827;
        --border: #1F2937;
        --text-main: #F9FAFB;
        --text-sub: #9CA3AF;
        --success: #22C55E;
        --warning: #F59E0B;
        --danger: #EF4444;
    }
    
    .stApp { background-color: var(--bg); color: var(--text-main); font-family: 'Inter', sans-serif; }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(124, 58, 237, 0.15);
        border-top: 1px solid rgba(124, 58, 237, 0.4);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3), inset 0 0 20px rgba(124, 58, 237, 0.05);
        margin-bottom: 20px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(124, 58, 237, 0.15);
    }

    /* Typography inside Cards */
    .card-title { color: var(--text-sub); font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;}
    .card-metric { color: var(--text-main); font-size: 2.2rem; font-weight: 800; margin: 0; line-height: 1.2;}
    .card-trend-up { color: var(--success); font-size: 0.85rem; font-weight: 600; display: flex; align-items: center; gap: 4px; margin-top: 5px;}
    .card-trend-down { color: var(--danger); font-size: 0.85rem; font-weight: 600; display: flex; align-items: center; gap: 4px; margin-top: 5px;}

    /* CRM Action Panel Styling */
    .crm-action-box {
        background: linear-gradient(145deg, #111827 0%, #1a1528 100%);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .action-btn-primary {
        background: var(--primary); color: white; border: none; padding: 6px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; cursor: pointer; width: 100%; margin-top: 10px;
    }
    
    /* Custom Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { color: var(--text-sub); font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary) !important; border-bottom-color: var(--primary) !important; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --------------------------------------------------
# 2. DATA MOCKING (For UI Display)
# --------------------------------------------------
@st.cache_data
def generate_premium_data():
    np.random.seed(42)
    users = [f"CUST-{np.random.randint(1000,9999)}" for _ in range(100)]
    probs = np.random.beta(0.5, 2, 100) # Right-skewed realistic probabilities
    df = pd.DataFrame({
        'Customer': users,
        'Risk Score': probs,
        'Tenure (M)': np.random.randint(1, 48, 100),
        'MRR ($)': np.random.randint(50, 500, 100),
        'Engagement': np.random.uniform(0.1, 1.0, 100),
        'Support Tickets': np.random.randint(0, 8, 100)
    })
    
    df['Segment'] = pd.cut(df['Risk Score'], bins=[0, 0.3, 0.7, 1.0], labels=['Low 🟢', 'Medium 🟡', 'High 🔴'])
    df['Top Driver'] = np.where(df['Support Tickets'] > 4, "Support Frustration", 
                       np.where(df['Engagement'] < 0.4, "Low App Usage", "Price Sensitivity"))
    return df.sort_values(by='Risk Score', ascending=False)

df = generate_premium_data()

# --------------------------------------------------
# 3. SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown(f"<h2 style='color:#7C3AED; font-weight:800; letter-spacing: -1px;'>⚡ ChurnOps</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9CA3AF; font-size:0.9rem;'>Predictive Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🔍 Global Filters")
    selected_segment = st.multiselect("Risk Segment", ['High 🔴', 'Medium 🟡', 'Low 🟢'], default=['High 🔴', 'Medium 🟡'])
    min_mrr = st.slider("Min Revenue (MRR)", 0, 500, 50)
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Config")
    st.selectbox("Active Model Engine", ["XGBoost v2.1 (Champion)", "Random Forest v1.0", "Logistic Reg (Baseline)"])
    st.caption("Last retrained: 2 hours ago")

filtered_df = df[(df['Segment'].isin(selected_segment)) & (df['MRR ($)'] >= min_mrr)]

# --------------------------------------------------
# 4. MAIN DASHBOARD LAYOUT
# --------------------------------------------------
st.markdown("<h2>Customer Churn Prediction System</h2>", unsafe_allow_html=True)

# ROW 1: KPI CARDS
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown("""
        <div class='glass-card'>
            <div class='card-title'>👥 Total Active Users</div>
            <div class='card-metric'>12,450</div>
            <div class='card-trend-up'>↗ +3.2% vs last month</div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown("""
        <div class='glass-card'>
            <div class='card-title'>📉 Projected Churn Rate</div>
            <div class='card-metric'>14.2%</div>
            <div class='card-trend-down'>↘ -1.5% vs last month</div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    high_risk = len(df[df['Segment'] == 'High 🔴'])
    st.markdown(f"""
        <div class='glass-card'>
            <div class='card-title'>⚠️ High Risk Cohort</div>
            <div class='card-metric' style='color: var(--danger);'>{high_risk}</div>
            <div class='card-trend-down'>↗ +12 accounts this week</div>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    rev_risk = df[df['Segment'] == 'High 🔴']['MRR ($)'].sum()
    st.markdown(f"""
        <div class='glass-card'>
            <div class='card-title'>💸 Revenue at Risk (MRR)</div>
            <div class='card-metric' style='color: var(--warning);'>${rev_risk:,}</div>
            <div class='card-trend-down'>Requires immediate action</div>
        </div>
    """, unsafe_allow_html=True)

# ROW 2: CHARTS
chart_col1, chart_col2 = st.columns([2, 1.2])

with chart_col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: var(--text-sub);'>📈 Weekly Churn Risk Trend</h4>", unsafe_allow_html=True)
    trend_x = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    trend_y = [18, 17.5, 16, 16.2, 15, 14.2]
    fig_area = go.Figure(go.Scatter(
        x=trend_x, y=trend_y, fill='tozeroy', mode='lines+markers',
        line=dict(color='#7C3AED', width=3), fillcolor='rgba(124, 58, 237, 0.2)', marker=dict(size=8, color='#A78BFA')
    ))
    fig_area.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0), height=250)
    st.plotly_chart(fig_area, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: var(--text-sub);'>📊 Risk Distribution</h4>", unsafe_allow_html=True)
    dist_df = df.groupby('Segment').size().reset_index(name='Count')
    fig_bar = px.bar(dist_df, x='Segment', y='Count', color='Segment', color_discrete_map={'High 🔴': '#EF4444', 'Medium 🟡': '#F59E0B', 'Low 🟢': '#22C55E'})
    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0), height=250, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ROW 3: EXPLAINABILITY & CRM ACTION PANEL
act_col1, act_col2 = st.columns([1.5, 1])

with act_col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: var(--text-sub);'>🧠 SHAP Explainability: Top Drivers</h4>", unsafe_allow_html=True)
    features = ['Support Frustration', 'Low App Usage', 'Price vs Tenure', 'Has Autopay']
    impact = [0.35, 0.28, 0.15, -0.22]
    colors = ['#EF4444', '#EF4444', '#F59E0B', '#22C55E']
    fig_shap = go.Figure(go.Bar(x=impact, y=features, orientation='h', marker_color=colors))
    fig_shap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0), height=250)
    st.plotly_chart(fig_shap, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with act_col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: var(--text-sub);'>🚨 Recommended Actions</h4>", unsafe_allow_html=True)
    top_3 = df[df['Segment'] == 'High 🔴'].head(3)
    for _, row in top_3.iterrows():
        action_text = "📞 Schedule Call" if row['Top Driver'] == "Support Frustration" else "✉️ Send Discount"
        st.markdown(f"""
        <div class='crm-action-box'>
            <div style='display:flex; justify-content:space-between;'>
                <b style='color:var(--text-main);'>{row['Customer']}</b>
                <span style='color:var(--danger); font-weight:bold;'>{row['Risk Score']*100:.0f}% Risk</span>
            </div>
            <div style='color:var(--text-sub); font-size:0.85rem; margin-top:4px;'>Driver: {row['Top Driver']}</div>
            <button class='action-btn-primary'>{action_text}</button>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ROW 4: THE 5 TABS (Including new Batch Upload)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Customer Intelligence Table", "🔍 Customer Drill-Down", "💰 Revenue Simulation", "⚖️ Model Comparison", "📁 Batch CSV Scoring"])

with tab1:
    display_df = filtered_df[['Customer', 'Risk Score', 'Segment', 'Top Driver', 'MRR ($)', 'Tenure (M)']].copy()
    display_df['Risk Score'] = (display_df['Risk Score'] * 100).round(1).astype(str) + '%'
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)

with tab2:
    st.markdown("### 🎯 Deep Dive Profile")
    selected_cust = st.selectbox("Select Customer to Analyze:", df[df['Segment'] == 'High 🔴']['Customer'].tolist())
    if selected_cust:
        cust_data = df[df['Customer'] == selected_cust].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score", f"{cust_data['Risk Score']*100:.1f}%", "- Critical", delta_color="inverse")
        c2.metric("Tenure", f"{cust_data['Tenure (M)']} Months")
        c3.metric("MRR", f"${cust_data['MRR ($)']}")
        st.markdown(f"**Primary Driver:** `{cust_data['Top Driver']}`")

with tab3:
    st.markdown("### 💰 ROI & Revenue Impact")
    target_rate = st.slider("Expected Retention Success Rate (%)", 5, 50, 15)
    total_risk_mrr = df[df['Segment'] == 'High 🔴']['MRR ($)'].sum()
    annual_saved = (total_risk_mrr * (target_rate / 100)) * 12
    st.markdown(f"""
    <div style='background:rgba(34, 197, 94, 0.1); border: 1px solid #22C55E; padding: 20px; border-radius:10px; text-align:center;'>
        <h3 style='color:#22C55E; margin:0;'>Predicted Annual Revenue Saved: ${annual_saved:,.2f}</h3>
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("### ⚖️ Champion vs Challenger")
    comp_data = pd.DataFrame({'Model': ['XGBoost', 'Random Forest', 'Logistic Reg'], 'ROC-AUC': [0.89, 0.82, 0.74], 'Recall': [0.88, 0.65, 0.45]})
    st.table(comp_data)

with tab5:
    st.markdown("### 📁 Batch Prediction (CSV Upload)")
    st.write("Upload a CSV file containing customer data to score hundreds of users at once.")
    uploaded_file = st.file_uploader("Upload Customer Data (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(batch_df)} rows.")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("🚀 Run AI Batch Scoring", type="primary"):
                with st.spinner("Scoring customers via FastAPI..."):
                    results = []
                    for index, row in batch_df.iterrows():
                        payload = {
                            "customer_id": str(row.get('customer_id', f"CUST-{index}")),
                            "tenure_months": int(row['tenure_months']),
                            "billing_amount": float(row['billing_amount']),
                            "support_tickets": int(row['support_tickets']),
                            "sla_breaches": int(row['sla_breaches']),
                            "active_days": int(row['active_days'])
                        }
                        try:
                            response = requests.post("http://127.0.0.1:8000/score", json=payload)
                            if response.status_code == 200:
                                results.append(response.json())
                        except Exception as e:
                            pass
                            
                    if results:
                        results_df = pd.DataFrame(results)
                        st.markdown("### 🎯 Scoring Complete")
                        st.dataframe(results_df, use_container_width=True)
                        csv_export = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Download Scored Data", data=csv_export, file_name="scored_customers.csv", mime="text/csv")
                    else:
                        st.error("Make sure your api.py is running on port 8000!")
        except Exception as e:
            st.error(f"Error processing file: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# 5. API TESTING PANEL
# --------------------------------------------------
with st.expander("🛠️ Developer: Test Real-Time FastAPI Endpoint"):
    st.write("Send a live payload to your XGBoost backend.")
    with st.form("api_test_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            test_tenure = st.number_input("Tenure", 1, 72, 12)
            test_billing = st.number_input("Billing ($)", 20.0, 150.0, 80.0)
        with col2:
            test_support = st.number_input("Support Tickets", 0, 20, 5)
            test_sla = st.number_input("SLA Breaches", 0, 10, 1)
        with col3:
            test_active = st.number_input("Active Days", 0, 30, 8)
            
        submit_api = st.form_submit_button("🚀 Predict")

    if submit_api:
        payload = {"customer_id": "TEST-99", "tenure_months": test_tenure, "billing_amount": test_billing, "support_tickets": test_support, "sla_breaches": test_sla, "active_days": test_active}
        try:
            response = requests.post("http://127.0.0.1:8000/score", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Response Received!")
                st.metric("Predicted Churn Risk", f"{result['churn_probability']*100:.1f}%", result['segment'], delta_color="inverse")
                st.json(result)
            else:
                st.error("API Error")
        except:
            st.error("❌ Could not connect. Is api.py running?")