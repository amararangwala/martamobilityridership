import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re

# MARTA Brand Colors
MARTA_BLUE = "#0092D0"
MARTA_GOLD = "#FDBE43"
MARTA_ORANGE = "#FF7500"

st.set_page_config(page_title="MARTA Mobility Client Condition Analysis", layout="wide")

st.title("MARTA Mobility Client Condition Analysis")

st.markdown("""
**How this works:**
1. Upload your client data file
2. Map the columns (including the condition tags column)
3. Optionally upload a separate conditions master list for validation
4. Score each condition tag by moving the sliders (1-3) 
6. Clients with multiple conditions receive the **MAX** score of their combined conditions
7. See real-time distribution as you adjust scores
""")

# -------------------------
# Upload: Clients FIRST (REQUIRED)
# -------------------------
st.subheader("1) Client Data File (Required)")
uploaded = st.file_uploader("Upload clients file (CSV or Excel)", type=["csv", "xlsx", "xls"], key="clients_file")

if uploaded is None:
    st.info("Please upload the **Clients** file to begin.")
    st.stop()

name = uploaded.name.lower()
is_excel = name.endswith(".xlsx") or name.endswith(".xls")

if is_excel:
    xls = pd.ExcelFile(uploaded)
    sheet_name = st.selectbox("Select sheet", xls.sheet_names, index=0)
    df = xls.parse(sheet_name)
else:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        df = pd.read_csv(uploaded, encoding="latin-1")

# -------------------------
# Column mapping for clients
# -------------------------
st.subheader("2) Map client columns")
lower_to_orig = {c.lower().strip(): c for c in df.columns}
all_cols = list(df.columns)

def pick(defaults, fallback_index=0):
    for cand in defaults:
        if cand.lower() in lower_to_orig:
            return lower_to_orig[cand.lower()]
    return all_cols[fallback_index] if all_cols else None

client_col_guess = pick(["client_id","id","client","customer_id"])
condition_col_guess = pick(["condition_code_description","condition","conditions","condition description","code description"])

client_col = st.selectbox("Client ID column", options=all_cols, index=all_cols.index(client_col_guess) if client_col_guess in all_cols else 0)
condition_col = st.selectbox("Condition code/description column", options=all_cols, index=all_cols.index(condition_col_guess) if condition_col_guess in all_cols else 0)

addr_mode = st.radio("Address input", options=["Single address column","Address components (GA)"], index=0)

if addr_mode == "Single address column":
    address_col = st.selectbox("Address column", options=["<none>"] + all_cols, index=0)
    street_col = on_street_col = city_col = zipcode_col = None
else:
    street_col = st.selectbox("Street number column", options=["<none>"] + all_cols, index=0)
    on_street_col = st.selectbox("On-street column", options=["<none>"] + all_cols, index=0)
    city_col = st.selectbox("City column", options=["<none>"] + all_cols, index=0)
    zipcode_col = st.selectbox("Zipcode column", options=["<none>"] + all_cols, index=0)
    address_col = None

# Normalization settings in sidebar
st.sidebar.header("Parsing Settings")
delims = st.sidebar.text_input("Delimiters that separate multiple tags", value=";,&|", help="Characters that separate multiple conditions")
normalize_case = st.sidebar.checkbox("Normalize tags to Title Case", value=True)
strip_punct = st.sidebar.checkbox("Strip punctuation (.-_)", value=True)
strip_quotes = st.sidebar.checkbox("Strip quotes", value=True)

def norm_tag(t: str) -> str:
    if t is None:
        return ""
    s = str(t).strip()
    if strip_quotes:
        s = s.replace('"', '').replace("'", '')
    if strip_punct:
        s = re.sub(r"[\._-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if normalize_case:
        s = s.title()
    return s

def split_tags(val: str, seps) -> list:
    if pd.isna(val) or str(val).strip() == "" or str(val).lower() in ["nan", "none"]:
        return ["(blank)"]
    s = str(val)
    for sep in seps:
        s = s.replace(sep, "|")
    s = re.sub(r"\s+\|\s+", "|", s)
    parts = [norm_tag(p) for p in s.split("|") if norm_tag(p) != ""]
    return parts if parts else ["(blank)"]

# -------------------------
# Conditions Master List (OPTIONAL)
# -------------------------
st.subheader("3) Conditions master list (optional)")

use_master_list = st.radio(
    "How do you want to define valid condition tags?",
    options=["Use all conditions from client data", "Upload separate conditions master list"],
    index=0
)

valid_tags = None
if use_master_list == "Use all conditions from client data":
    st.info("Using all unique conditions found in the client data file.")
    # Extract all unique conditions from the client data
    work_temp = df.copy()
    work_temp.rename(columns={condition_col: "condition"}, inplace=True)
    
    seps = [s for s in [s.strip() for s in delims.split(",")] if s]
    if " and" not in seps and "and" not in seps:
        seps.append(" and ")
    
    all_conditions = []
    for val in work_temp["condition"].dropna():
        tags = split_tags(val, seps)
        all_conditions.extend(tags)
    
    valid_tags = sorted({t for t in all_conditions if t != ""})
    if "(blank)" not in valid_tags:
        all_tags = ["(blank)"] + valid_tags
    else:
        all_tags = valid_tags
    
    st.success(f"Found {len(valid_tags)} unique condition tags in your data.")
    
else:  # Upload separate master list
    cond_file = st.file_uploader("Upload a CSV/XLSX that lists valid single-condition tags", type=["csv","xlsx","xls"], key="cond_master")
    
    if cond_file is not None:
        try:
            if cond_file.name.lower().endswith((".xlsx",".xls")):
                x = pd.ExcelFile(cond_file)
                sheet_opt = st.selectbox("Master list sheet", x.sheet_names, index=0, key="cond_sheet")
                d = x.parse(sheet_opt)
            else:
                try:
                    d = pd.read_csv(cond_file)
                except Exception:
                    d = pd.read_csv(cond_file, encoding="latin-1")
            cond_col = st.selectbox("Column containing condition names", options=list(d.columns), index=0, key="cond_col_pick")
            
            # Build the master tag universe (normalized)
            valid_tags = sorted({norm_tag(x) for x in d[cond_col].dropna().astype(str) if norm_tag(x) != ""})
            if "(blank)" not in valid_tags:
                all_tags = ["(blank)"] + valid_tags
            else:
                all_tags = valid_tags
            
            st.success(f"Loaded {len(valid_tags)} condition tags from master list.")
            
        except Exception as e:
            st.error(f"Could not read conditions file: {e}")
            st.stop()
    else:
        st.info("Please upload the conditions master list file.")
        st.stop()

# Compose GA address if using components
def compose_ga_address(row):
    left = " ".join([str(x).strip() for x in [
        row.get(street_col) if street_col and street_col != "<none>" else "",
        row.get(on_street_col) if on_street_col and on_street_col != "<none>" else ""
    ] if str(x).strip() != ""])
    right = ", ".join([str(x).strip() for x in [
        row.get(city_col) if city_col and city_col != "<none>" else "",
        "GA"
    ] if str(x).strip() != ""])
    tail = str(row.get(zipcode_col)).strip() if zipcode_col and zipcode_col != "<none>" and pd.notna(row.get(zipcode_col)) else ""
    addr = left
    if right:
        addr = f"{addr}, {right}" if addr else right
    if tail:
        addr = f"{addr} {tail}" if addr else tail
    return addr

# Build working frame
work = df.copy()
work.rename(columns={client_col: "client_id", condition_col: "condition"}, inplace=True)
if addr_mode == "Single address column":
    if address_col and address_col != "<none>":
        work.rename(columns={address_col: "address"}, inplace=True)
    else:
        work["address"] = ""
else:
    work["address"] = work.apply(compose_ga_address, axis=1)

# Preserve raw city/zip if available (for group-by later)
_raw_city = work[city_col].copy() if addr_mode == "Address components (GA)" and city_col and city_col != "<none>" else None
_raw_zip = work[zipcode_col].copy() if addr_mode == "Address components (GA)" and zipcode_col and zipcode_col != "<none>" else None

# Aggregate rows per client (merge multiple condition cells)
agg = (work.groupby(["client_id","address"], dropna=False)["condition"]
          .apply(lambda s: "; ".join([str(x) for x in s if str(x).strip() != ""]))
          .reset_index())

# Split + normalize
seps = [s for s in [s.strip() for s in delims.split(",")] if s]
if " and" not in seps and "and" not in seps:
    seps.append(" and ")
agg["tags_list_raw"] = agg["condition"].apply(lambda v: split_tags(v, seps))

# Whitelist to master tags and collect unmatched for QA
valid_set = set(valid_tags)
unmatched = set()
def keep_valid(tags):
    kept = []
    for t in tags:
        if t == "(blank)" or t in valid_set:
            kept.append(t)
        else:
            unmatched.add(t)
    return kept if kept else ["(blank)"]
agg["tags_list"] = agg["tags_list_raw"].apply(keep_valid)

# Show unmatched tags if using master list
if use_master_list == "Upload separate conditions master list" and unmatched:
    st.warning(f"Found {len(unmatched)} condition tags in your data that don't match the master list. These will be excluded from scoring.")
    with st.expander("View unmatched tags"):
        st.write(sorted(unmatched))

# -------------------------
# Tag scoring UI
# -------------------------
st.subheader("4) Assign scores to condition tags (1–3)")
st.markdown("""Score 1 = Light Dependency on Curb-to-Curb service; good candidate for Feeder service  
Score 3 = High Dependency on Curb-to-Curb service; unlikely candidate for Feeder service  
**Blank/null conditions automatically get a score of 3 - High Dependency** (unconditional eligibility for Curb-to-Curb service)""")
tag_filter = st.text_input("Filter tags (case-insensitive contains)", value="").strip().lower()

cols = st.columns(3)
tag_scores = {}
for i, tag in enumerate(all_tags):
    if tag_filter and tag_filter not in tag.lower():
        continue
    with cols[i % 3]:
        tag_scores[tag] = st.select_slider(
            label=tag.replace("(blank)", "Unconditional"),
            options=[1,2,3],
            value=3 if tag == "(blank)" else 2,
            help="1 = low priority, 3 = high priority"
        )

# Score by MAX across a client's tags
def max_score(tags):
    vals = [tag_scores.get(t, 2) for t in tags] if tags else [tag_scores.get("(blank)", 3)]
    return int(np.max(vals))

agg["category"] = agg["tags_list"].apply(max_score)
agg["is_unconditional"] = agg["tags_list"].apply(lambda ts: set(ts) == {"(blank)"} )

# Distribution table
dist = (agg["category"].value_counts()
        .reindex([1,2,3], fill_value=0)
        .rename_axis("category")
        .reset_index(name="count"))
total = int(len(agg))
dist["percent"] = (dist["count"]/total*100).round(2)

left, right = st.columns(2)
with left:
    st.markdown("### Distribution by Category")
    st.dataframe(dist, use_container_width=True)
with right:
    st.markdown("### % of Clients by Category (Bar)")
    bar = (alt.Chart(dist)
             .mark_bar()
             .encode(
                 x=alt.X("category:O", title="Category (1–3)"),
                 y=alt.Y("percent:Q", title="% of Clients"),
                 color=alt.Color("category:O", 
                                scale=alt.Scale(domain=[1,2,3], range=[MARTA_BLUE, MARTA_GOLD, MARTA_ORANGE]),
                                legend=None),
                 tooltip=["category","count",alt.Tooltip("percent:Q", format=".2f")]
             )
             .properties(height=320))
    st.altair_chart(bar, use_container_width=True)

# Pie chart
pie_df = pd.DataFrame({
    "slice": ["Unconditional","Low Dependency","Medium Dependency","High Dependency"],
    "count": [
        int(agg["is_unconditional"].sum()),
        int(((agg["category"]==1) & (~agg["is_unconditional"])).sum()),
        int(((agg["category"]==2) & (~agg["is_unconditional"])).sum()),
        int(((agg["category"]==3) & (~agg["is_unconditional"])).sum()),
    ]
})
pie_df["percent"] = (pie_df["count"]/total*100).round(2)

st.markdown("### Unconditional vs Scores 1–3 (Pie)")
pie = (alt.Chart(pie_df)
         .mark_arc(innerRadius=60)
         .encode(
             theta=alt.Theta(field="count", type="quantitative"),
             color=alt.Color(field="slice", type="nominal", 
                            scale=alt.Scale(domain=["Unconditional","Low Dependency","Medium Dependency","High Dependency"],
                                          range=[MARTA_ORANGE, MARTA_BLUE, MARTA_GOLD, "#CCCCCC"]),
                            legend=alt.Legend(title="Group")),
             tooltip=[alt.Tooltip("slice:N", title="Group"),
                      alt.Tooltip("count:Q", title="Clients"),
                      alt.Tooltip("percent:Q", title="% of all clients", format=".2f")]
         )
         .properties(height=320))
st.altair_chart(pie, use_container_width=True)

# --------- Optional stacked bar by City/Zip ---------
group_choice = None
if _raw_city is not None or _raw_zip is not None:
    st.markdown("### Breakdown by geography (optional)")
    opt = st.radio("Group by", options=["None"] + [opt for opt in ["City","Zip"] if (opt=="City" and _raw_city is not None) or (opt=="Zip" and _raw_zip is not None)], index=0, horizontal=True)
    group_choice = None if opt == "None" else opt

if group_choice is not None:
    geo = df[[client_col] + ([city_col] if group_choice=="City" else [zipcode_col])].copy()
    geo = geo.rename(columns={client_col:"client_id", (city_col if group_choice=="City" else zipcode_col):"geo"})
    geo["geo"] = geo["geo"].astype(str)
    geo = geo.groupby("client_id", dropna=False)["geo"].agg(lambda s: next((x for x in s if str(x).strip() != "" and pd.notna(x)), "")).reset_index()
    merged = pd.merge(agg[["client_id","category"]], geo, on="client_id", how="left")
    merged["geo"] = merged["geo"].fillna("")
    top = merged["geo"].value_counts().head(10).index.tolist()
    merged["geo_bucket"] = merged["geo"].apply(lambda g: g if g in top and g != "" else ("Other" if g != "" else "Unknown"))
    stacked = merged.groupby(["geo_bucket","category"]).size().reset_index(name="count")
    stacked["percent"] = stacked.groupby("geo_bucket")["count"].transform(lambda s: (s / s.sum() * 100).round(2))
    st.caption("Top 10 groups shown; others bucketed as 'Other'.")
    stacked_chart = (alt.Chart(stacked)
                      .mark_bar()
                      .encode(
                          x=alt.X("geo_bucket:N", title=group_choice),
                          y=alt.Y("percent:Q", title="% within group"),
                          color=alt.Color("category:O", title="Category",
                                        scale=alt.Scale(domain=[1,2,3], range=[MARTA_BLUE, MARTA_GOLD, MARTA_ORANGE])),
                          tooltip=["geo_bucket","category","count",alt.Tooltip("percent:Q", format=".2f")]
                      )
                      .properties(height=360))
    st.altair_chart(stacked_chart, use_container_width=True)

st.markdown("---")
st.markdown("### Drilldown")
sel = st.selectbox("Show clients in category", options=[1,2,3], index=2)
st.dataframe(
    agg.loc[agg["category"]==sel, ["client_id","address","condition","tags_list","category"]],
    use_container_width=True
)

# Unmatched tag QA (only show at bottom if using "all conditions from data")
if use_master_list == "Use all conditions from client data" and unmatched:
    with st.expander("Tags that did not match your master list"):
        st.write(sorted(unmatched))

# Export
out_csv = agg.to_csv(index=False).encode("utf-8")
st.download_button("Download scored clients (CSV)", data=out_csv, file_name="scored_clients.csv", mime="text/csv")

st.success(f"Processed {total} clients with {len(all_tags)} condition tags.")
