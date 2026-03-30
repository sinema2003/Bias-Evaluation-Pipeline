import os
import json
import pandas as pd
import streamlit as st
from urllib.request import Request, urlopen

st.set_page_config(page_title="Bias Eval Dashboard", layout="wide")


def _load_dotenv_if_present() -> None:
    """
    Minimal .env loader (no external deps).
    Sets os.environ entries for KEY=VALUE pairs.
    """
    # Prefer repo-root .env (next to app/ and src/).
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                key, val = s.split("=", 1)
                key = key.strip()
                # Remove surrounding quotes if present (common .env style).
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ and val != "":
                    os.environ[key] = val
    except Exception:
        # Don't block dashboard UI if .env is missing or unreadable.
        return


_load_dotenv_if_present()

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

    st.divider()
    st.subheader("AI summary (Gemini)")
    gemini_model = st.selectbox(
        "Gemini model",
        ["gemini-3-flash", "gemini-2.5-pro", "gemini-2.5-flash"],
        index=0,
        key="gemini_model",
    )
    max_chars = st.slider(
        "Max chars to send (truncates for context limits)",
        min_value=0,
        max_value=50000,
        value=50000,
        step=1000,
        key="gemini_max_chars",
    )

    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        gemini_api_key = st.text_input(
            "GEMINI_API_KEY (or set env var GEMINI_API_KEY)",
            type="password",
            key="gemini_api_key",
        )

    if st.button("AI summarize enriched jsonl", key="gemini_summarize_btn"):
        if not gemini_api_key:
            st.error("Missing GEMINI_API_KEY. Set it in your environment or enter it above.")
        else:
            with st.spinner("Sending enriched jsonl to Gemini..."):
                try:
                    with open(enriched_path, "r", encoding="utf-8") as f:
                        enriched_text = f.read()
                except Exception as e:
                    st.error(f"Failed to read enriched jsonl: {e}")
                    enriched_text = ""

                if enriched_text:
                    if max_chars and len(enriched_text) > max_chars:
                        enriched_text_send = enriched_text[:max_chars]
                        trunc_note = (
                            f"\n\nNOTE: Input was truncated to first {max_chars} characters "
                            f"(total file length={len(enriched_text)})."
                        )
                    else:
                        enriched_text_send = enriched_text
                        trunc_note = ""

                    prompt = (
                        "You are summarizing a bias evaluation run.\n\n"
                        "Given the following enriched.jsonl content, produce a concise summary "
                        "covering:\n"
                        "1) Which categories are present and what stands out at a high level.\n"
                        "2) Overall toxicity and sentiment tendencies.\n"
                        "3) Any notable outliers (e.g., unexpectedly high toxicity).\n"
                        "4) Differences between 'stereotype' vs 'anti_stereotype' where obvious.\n\n"
                        "Return the summary as short bullet points."
                        f"{trunc_note}\n\n"
                        "enriched.jsonl:\n"
                        f"{enriched_text_send}"
                    )

                    endpoint = (
                        "https://generativelanguage.googleapis.com/v1beta/models/"
                        f"{gemini_model}:generateContent?key={gemini_api_key}"
                    )
                    payload = {
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 250000},
                    }
                    req = Request(
                        endpoint,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    try:
                        resp = urlopen(req, timeout=120)
                        data = json.loads(resp.read().decode("utf-8"))
                    except Exception as e:
                        st.error(f"Gemini request failed: {e}")
                        data = None

                    if data:
                        from pprint import pprint
                        print(pprint(data))
                        try:
                            summary_text = (
                                data.get("candidates", [{}])[0]
                                .get("content", {})
                                .get("parts", [{}])[0]
                                .get("text", "")
                            )
                        except Exception:
                            summary_text = ""

                        if summary_text.strip():
                            st.markdown(summary_text)
                        else:
                            st.error(
                                "Gemini returned no text. Check API key/model permissions and retry."
                            )
                else:
                    st.error("enriched.jsonl file was empty or could not be read.")
else:
    st.info("No enriched jsonl found. Run `aggregate.py` first.")