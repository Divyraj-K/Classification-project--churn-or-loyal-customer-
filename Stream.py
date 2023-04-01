import streamlit as st
import pandas as pd


st.write("Divyraj is mad")
st.title("Divyraj is joker")
st.sidebar.title("Divyraj is idiot")

friends = ["s","d","z","m","p"]
df = pd.DataFrame(friends)
st.table(df)
st.dataframe(df)

