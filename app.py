import streamlit as st
import os
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# --- STREAMLIT FRONTEND SETUP ---
st.set_page_config(page_title="AI Equity Dashboard", layout="wide", page_icon="📈")
st.title("🤖 Free Multi-Agent LLM Share Market Analyst")
st.caption("Instant data visualization panels combined with custom multi-agent sequential context reasoning.")

# Configuration Panel on the left side of the dashboard
st.sidebar.header("🔑 Configuration")
groq_api_key = st.sidebar.text_input("Enter Groq API Key:", type="password", help="Get a free key from ://groq.com")
selected_model = st.sidebar.selectbox("Choose Free Model:", ["llama3-70b-8192", "mixtral-8x7b-32768"])

# --- AI AGENT BACKEND INTERNALS ---
web_search_tool = DuckDuckGoSearchRun()

@tool("Fetch Stock Fundamentals")
def fetch_stock_fundamentals(ticker: str) -> str:
    """Fetches key fundamental ratios for a given stock ticker from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        metrics = {
            "Company Name": info.get("longName"),
            "Current Price": info.get("currentPrice"),
            "Trailing P/E": info.get("trailingPE"),
            "Debt to Equity": info.get("debtToEquity"),
            "Return on Equity (ROE)": info.get("returnOnEquity"),
            "Operating Margin": info.get("operatingMargins"),
            "Free Cash Flow": info.get("freeCashflow")
        }
        return "\n".join([f"{k}: {v}" for k, v in metrics.items()])
    except Exception as e:
        return f"Error fetching metrics for {ticker}: {str(e)}"

def run_ai_analysis(ticker, api_key, model_name):
    """Executes specialized sequential multi-agent collaborative workflows."""
    llm = ChatGroq(temperature=0.1, groq_api_key=api_key, model_name=model_name)
    
    financial_analyst = Agent(
        role="Senior Fundamental Analyst",
        goal="Evaluate financial structural patterns using quantitative ratios.",
        backstory="You are an expert value investor reviewing multiples, balance sheets, and leverage states.",
        tools=[fetch_stock_fundamentals],
        llm=llm,
        verbose=True
    )

    news_scout = Agent(
        role="Macro Risk & Geopolitical Analyst",
        goal="Identify geopolitical events, macroeconomic tailwinds, or business deals affecting a stock.",
        backstory="You track international headlines, trade shifts, raw material issues, and corporate deals.",
        tools=[web_search_tool],
        llm=llm,
        verbose=True
    )

    portfolio_manager = Agent(
        role="Investment Committee Chair",
        goal="Synthesize financials against macro real-time headlines to make tactical Buy/Sell decisions.",
        backstory="You are an elite fund allocator who cuts through speculative noise to guard asset capital.",
        llm=llm,
        verbose=True
    )

    task_fundamentals = Task(
        description=f"Evaluate key fundamental safety factors and margin parameters for ticker: {ticker}.",
        expected_output="A clean textual summary profiling structural value and leverage data.",
        agent=financial_analyst
    )

    task_news = Task(
        description=f"Scan for active global structural news events, wars, macro developments, or major commercial deals relative to: {ticker}.",
        expected_output="An outline profiling macro sentiment directions and near-term market events.",
        agent=news_scout
    )

    task_decision = Task(
        description=f"Synthesize outputs from previous tasks for {ticker} to produce a high-conviction decision profile.",
        expected_output="A layout starting with 'RECOMMENDATION: [BUY / SELL / HOLD]' followed by 3 core validation bullet points.",
        agent=portfolio_manager
    )

    crew = Crew(
        agents=[financial_analyst, news_scout, portfolio_manager],
        tasks=[task_fundamentals, task_news, task_decision],
        process=Process.sequential
    )
    
    return crew.kickoff()

# --- DATAVIZ PANEL DEFINITION ---
def build_market_dashboard(ticker_symbol):
    """Fetches real-time market assets to build structural analytical charts."""
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Panel A: Numerical Parameter Metric Cards
        st.subheader(f"📈 Strategic Parameters Dashboard: {info.get('longName', ticker_symbol)}")
        m1, m2, m3, m4 = st.columns(4)
        
        currency = info.get("currency", "$")
        m1.metric("Current Price", f"{currency} {info.get('currentPrice', 'N/A')}")
        m2.metric("Trailing P/E Ratio", info.get("trailingPE", "N/A"))
        
        de_val = info.get("debtToEquity")
        m3.metric("Debt-to-Equity (D/E)", f"{de_val / 100:.2f}" if de_val else "N/A")
        
        roe_val = info.get("returnOnEquity")
        m4.metric("Return on Equity (ROE)", f"{roe_val * 100:.2f}%" if roe_val else "N/A")
        
        # Panel B: Multi-variable Interactive Layout (Candlestick Price + Volume Bars)
        st.write("---")
        st.subheader("📊 Stock Chart Analytics (1-Year Candlestick & Volume Dynamics)")
        
        hist_data = stock.history(period="1y")
        if not hist_data.empty:
            # Build an layout container grid containing two distinct row scales
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, subplot_titles=('Price Trend', 'Volume Traded'), 
                                row_width=[0.3, 0.7])
            
            # Subplot 1: Candlesticks
            fig.add_trace(go.Candlestick(
                x=hist_data.index, open=hist_data['Open'], high=hist_data['High'],
                low=hist_data['Low'], close=hist_data['Close'], name="Candlestick"
            ), row=1, col=1)
            
            # Subplot 2: Bar Volume
            fig.add_trace(go.Bar(
                x=hist_data.index, y=hist_data['Volume'], name="Volume", marker_color='orange'
            ), row=2, col=1)
            
            fig.update_layout(
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                margin=dict(l=30, r=30, t=30, b=30),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Historical data arrays unavailable for this specific tracker handle.")
            
        return True
    except Exception as e:
        st.error(f"Unable to load visualization panels: {e}")
        return False

# --- FRONTEND ROUTING CONTROL ---
ticker_input = st.text_input("Enter Share Ticker Symbol (e.g., AAPL, TSLA, INFY.NS):", value="AAPL").upper()

if ticker_input:
    valid_render = build_market_dashboard(ticker_input)
    st.write("---")
    st.subheader("🧠 Multi-Agent Analytical Consultation Panel")
    
    if st.button("Run Multi-Agent Macro Analysis"):
        if not groq_api_key:
            st.error("Please configure your free Groq API key inside the left sidebar setup pane.")
        elif valid_render:
            with st.spinner(f"AI Agents are evaluating {ticker_input} against macro news updates... Please wait."):
                try:
                    final_verdict = run_ai_analysis(ticker_input, groq_api_key, selected_model)
                    st.success("AI Synthesis Pipeline Completed!")
                    st.subheader("📋 Final Investment Committee Report")
                    st.markdown(final_verdict)
                except Exception as e:
                    st.error(f"An exception occurred within agent framework orchestration lifecycle: {e}")
