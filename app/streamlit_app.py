import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bias Eval Dashboard", layout="wide")

st.title("Bias Evaluation Pipeline - Dashboard")

summ_dir = "outputs/summaries"
runs = sorted([f for f in os.listdir(summ_dir) if f.endswith(".summary.csv")]) if os.path.isdir(summ_dir) else []
if not runs:
    st.warning("No summaries found. Run aggregation first.")
    st.stop()

pick = st.selectbox("Select run summary", runs)
run_id = pick.replace(".summary.csv", "")
summary_path = os.path.join(summ_dir, pick)
gaps_path = os.path.join(summ_dir, f"{run_id}.gaps.csv")
enriched_path = os.path.join(summ_dir, f"{run_id}.enriched.jsonl")

summary = pd.read_csv(summary_path)
gaps = pd.read_csv(gaps_path) if os.path.exists(gaps_path) else pd.DataFrame()

c1, c2 = st.columns(2)
with c1:
    st.subheader("Summary (group stats)")
    st.dataframe(summary, use_container_width=True, height=420)
with c2:
    st.subheader("Bias gaps (max-min across attributes)")
    if len(gaps):
        st.dataframe(gaps, use_container_width=True, height=420)
    else:
        st.info("No gaps computed (need >=2 attributes per category).")

st.divider()
st.subheader("Explore generations (enriched)")
if os.path.exists(enriched_path):
    # streamlit can read jsonl via pandas
    df = pd.read_json(enriched_path, lines=True)
    cat = st.selectbox("Category", sorted(df["category"].unique()))
    dec = st.selectbox("Decoding", sorted(df["decoding"].unique()))
    filt = df[(df["category"] == cat) & (df["decoding"] == dec)]
    st.write(f"Rows: {len(filt)}")
    st.dataframe(
        filt[["prompt_id","attribute","prompt","generation","toxicity","sentiment"]],
        use_container_width=True,
        height=520
    )
else:
    st.info("No enriched jsonl found. Run `aggregate.py` first.")