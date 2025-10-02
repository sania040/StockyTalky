import streamlit as st

def load_css():
    """Load all custom CSS styles"""
    st.markdown("""    
    <style>
    .main { background-color: #f5f7f9; }
    .stApp { max-width: 1200px; margin: 0 auto; }
    
    /* Card styles */
    .card {
        border-radius: 5px;
        padding: 1rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Metric styles */
    .metric-value {color: #333; font-size: 1.5rem; font-weight: bold;}
    .metric-label {color: #666; font-size: 0.9rem;}
    .positive {color: #4CAF50;}
    .negative {color: #F44336;}
    
    /* Recommendation styles */
    .buy {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        padding-left: 1rem;
    }
    .sell {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #F44336;
        padding-left: 1rem;
    }
    .hold {
        background-color: rgba(117, 117, 117, 0.1);
        border-left: 4px solid #757575;
        padding-left: 1rem;
    }
    
    /* Header styles */
    .crypto-header {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
    }
    
    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        padding: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    
    div.block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #1E3A8A; }
    </style>
    """, unsafe_allow_html=True)