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
st.subheader("Prompt distribution by category")

cat_counts = (
    summary.groupby("category", as_index=False)["n"]
    .sum()
    .sort_values("n", ascending=False)
)

if len(cat_counts):
    d1, d2 = st.columns(2)
    with d1:
        st.caption("Bar chart: number of prompt samples per category")
        st.bar_chart(cat_counts.set_index("category")["n"], use_container_width=True)
    with d2:
        st.caption("Pie chart: share of prompts by category")
        pie = cat_counts.rename(columns={"n": "count"})
        st.pyplot(
            pie.set_index("category")["count"].plot.pie(
                autopct="%1.1f%%",
                ylabel="",
                figsize=(6, 6),
            ).get_figure()
        )
else:
    st.info("No category counts found in summary.")

st.divider()
st.subheader("Model progress across runs")

all_run_rows = []
for run_file in runs:
    rid = run_file.replace(".summary.csv", "")
    run_path = os.path.join(summ_dir, run_file)
    try:
        run_df = pd.read_csv(run_path)
        run_df["run_id"] = rid
        all_run_rows.append(run_df)
    except Exception:
        continue

if all_run_rows:
    hist = pd.concat(all_run_rows, ignore_index=True)
    model_pick = st.selectbox(
        "Model for trend view",
        sorted(hist["model"].dropna().unique()),
        key="trend_model",
    )
    metric_pick = st.selectbox(
        "Metric",
        ["tox_mean", "sent_mean", "n"],
        index=0,
        key="trend_metric",
    )
    decoding_pick = st.selectbox(
        "Decoding",
        ["(all)"] + sorted(hist["decoding"].dropna().unique()),
        key="trend_decoding",
    )
    category_pick = st.selectbox(
        "Category",
        ["(all)"] + sorted(hist["category"].dropna().unique()),
        key="trend_category",
    )

    trend = hist[hist["model"] == model_pick].copy()
    if decoding_pick != "(all)":
        trend = trend[trend["decoding"] == decoding_pick]
    if category_pick != "(all)":
        trend = trend[trend["category"] == category_pick]

    if len(trend):
        trend["run_time"] = pd.to_datetime(trend["run_id"], format="%Y%m%d_%H%M%S", errors="coerce")
        trend = trend.sort_values(["run_time", "run_id"])
        trend_agg = trend.groupby("run_id", as_index=False)[metric_pick].mean()
        st.line_chart(trend_agg.set_index("run_id")[metric_pick], use_container_width=True)
        st.caption(
            f"{metric_pick} averaged over selected filters for model '{model_pick}'."
        )
    else:
        st.info("No rows match current model/filter selection.")
else:
    st.info("No run summaries could be loaded for trend chart.")

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