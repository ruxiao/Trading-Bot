import streamlit as st

# Set page config
st.set_page_config(page_title="Trading Bot Test", layout="wide")

# Main title
st.title("Trading Bot Test App")

# Add some text
st.write("This is a simple test to see if Streamlit is working correctly.")

# Add a button
if st.button("Click Me"):
    st.success("Button clicked!")

# Add a sidebar
st.sidebar.title("Controls")
st.sidebar.write("This is a sidebar")

# Display some data
import pandas as pd
import numpy as np

# Create random data
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=10),
    'Price': np.random.randn(10).cumsum() + 100
})

# Display a chart
st.subheader("Sample Price Chart")
st.line_chart(data.set_index('Date'))