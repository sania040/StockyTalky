import streamlit as st
from src.utils.styles import load_css
from src.pages import data_collection, dashboard, ml_recommendations, advanced_analytics, chatbot

# Must be first Streamlit command
st.set_page_config(
    page_title="StockyTalky - Crypto Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Load styles
    load_css()
    
    # Sidebar navigation
    st.sidebar.title("StockyTalky")
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/investment.png", width=100)
    
    page = st.sidebar.radio(
        "Navigation", 
        [
            "Data Collection",
            "Investment Dashboard",
            "ML Recommendations",
            "Advanced Analytics",
            "Crypto Assistant"
        ]
    )
    
    # Route to appropriate page
    if page == "Data Collection":
        data_collection.show()
    elif page == "Investment Dashboard":
        dashboard.show()
    elif page == "ML Recommendations":
        ml_recommendations.show()
    elif page == "Advanced Analytics":
        advanced_analytics.show()
    elif page == "Crypto Assistant":
        chatbot.show()

if __name__ == "__main__":
    main()