# src/utils/styles.py (or wherever your load_css function is)
import streamlit as st

def load_css():
    """Load improved custom CSS styles"""
    st.markdown("""
    <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* --- Global Styles --- */
        body {
            font-family: 'Inter', sans-serif;
            background-color: #F8FAFC; /* Lighter background */
            color: #374151; /* Darker gray text */
        }
        .stApp {
            max-width: 1300px; /* Slightly wider */
            margin: 0 auto;
        }
        div.block-container {
            padding: 2rem 2.5rem 3rem 2.5rem; /* More padding */
        }

        /* --- Typography --- */
        h1, h2, h3, h4, h5, h6 {
            color: #1E3A8A; /* Consistent header blue */
            font-weight: 600;
        }
        h1 {
            border-bottom: 2px solid #DBEAFE; /* Light blue underline */
            padding-bottom: 0.75rem;
            margin-bottom: 1.5rem;
        }
        h2 {
            margin-top: 2.5rem;
            margin-bottom: 1rem;
        }
        h3 {
             margin-top: 1.5rem;
             margin-bottom: 0.75rem;
             font-size: 1.4rem;
        }
        p, .stText, .stMarkdown {
            line-height: 1.6;
        }

        /* --- Sidebar --- */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E5E7EB;
        }
        [data-testid="stSidebar"] h1 {
            color: #1E3A8A;
            font-size: 1.8rem;
            border-bottom: none; /* Remove underline in sidebar */
        }
        [data-testid="stSidebar"] .stRadio > label > div[role="radiogroup"] > label {
            padding: 0.5rem 0.75rem; /* Add padding to radio items */
            border-radius: 8px;
            transition: background-color 0.2s ease;
        }
        [data-testid="stSidebar"] .stRadio > label > div[role="radiogroup"] > label:hover {
             background-color: #F3F4F6; /* Light hover effect */
        }


        /* --- Main Content Styling --- */
        .card {
            border-radius: 12px; /* Softer radius */
            padding: 1.5rem;
            background-color: #FFFFFF;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.04); /* Softer shadow */
            margin-bottom: 1.5rem;
            border: 1px solid #E5E7EB; /* Subtle border */
            transition: box-shadow 0.2s ease;
        }
        .card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -2px rgba(0, 0, 0, 0.05); /* Slightly stronger hover shadow */
        }

        /* --- Metric Cards --- */
        /* Apply this class using markdown: st.markdown("<div class='metric-card'>", unsafe_allow_html=True) ... st.markdown("</div>", unsafe_allow_html=True) */
        .metric-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
            border: 1px solid #E5E7EB;
            margin-bottom: 1.5rem;
            transition: box-shadow 0.2s ease;
        }
        .metric-card:hover {
             box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        .metric-card .stMetric { /* Style Streamlit's metric inside card */
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        .metric-card .stMetric > label { /* Label style */
            font-weight: 500;
            color: #6B7280; /* Gray label */
        }
        .metric-card .stMetric > div { /* Value style */
            font-size: 2rem;
            font-weight: 600;
            color: #111827; /* Dark value */
        }
        .metric-card .stMetric > div[data-testid="metric-delta"] { /* Delta style */
            font-size: 1rem;
            font-weight: 500;
        }

        /* Positive/Negative Colors */
        .positive { color: #10B981; } /* Green */
        .negative { color: #EF4444; } /* Red */

        /* Recommendation/Status styles */
        .buy { border-left: 5px solid #10B981; }
        .sell { border-left: 5px solid #EF4444; }
        .hold { border-left: 5px solid #6B7280; }

        /* --- Header Styles --- */
        .crypto-header {
            background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
        }
        .crypto-header h1 {
            color: white;
            font-size: 2.2rem;
            margin: 0;
            border: none;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }
        .crypto-header p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }

        /* --- Tab Styles --- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px; /* Spacing between tabs */
            border-bottom: 2px solid #E5E7EB; /* Underline for the tab bar */
        }
        .stTabs [data-baseweb="tab"] {
            height: auto; /* Allow height to adjust */
            background-color: transparent; /* No background for inactive tabs */
            border-radius: 0;
            padding: 0.75rem 0.5rem; /* Adjust padding */
            margin-bottom: -2px; /* Align with bottom border */
            border-bottom: 2px solid transparent; /* Placeholder border */
            color: #6B7280; /* Gray inactive text */
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: transparent;
            color: #1E3A8A; /* Blue active text */
            border-bottom: 2px solid #2563EB; /* Blue underline for active tab */
            font-weight: 600;
        }

        /* --- Button Styles --- */
        .stButton>button {
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            background-color: #2563EB; /* Primary blue */
            color: white;
            border: none;
            transition: background-color 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #1D4ED8; /* Darker blue */
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
         .stButton>button:active {
             background-color: #1E40AF;
         }

        /* --- Input Styles --- */
         .stTextInput>div>div>input, .stSelectbox>div>div>div {
             border-radius: 8px !important;
             border: 1px solid #D1D5DB !important;
         }
         .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus-within {
             border-color: #2563EB !important;
             box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
         }

    </style>
    """, unsafe_allow_html=True)

# You would call load_css() in your main app.py file
# e.g.,
# if __name__ == "__main__":
#     load_css()
#     main()