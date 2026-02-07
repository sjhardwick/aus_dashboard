"""
Australian Macroeconomic Dashboard
Fetches data from the ABS API and creates interactive charts for:
- GDP contributions by expenditure component
- Current account balance by component
- Inflation (CPI)
- Labour market indicators
"""

import io
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

# ABS API Configuration
ABS_API_BASE = "https://data.api.abs.gov.au/rest/data"

# GDP expenditure components
GDP_COMPONENTS = {
    "FCE": "Consumption",
    "GFC": "GFCF",
    "IST": "Inventories",
    "XGS": "Exports",
    "MGS": "Imports",
}

# Current account components
CA_COMPONENTS = {
    170: "Goods",
    180: "Services",
    8700: "Primary Income",
    8100: "Secondary Income",
}

# Labour force measures
LF_MEASURES = {
    "M13": "Unemployment Rate",
    "M12": "Participation Rate",
    "M16": "Employment/Pop Ratio",
}

# Colors for GDP components
GDP_COLORS = {
    "Consumption": "#4C78A8",   # Blue
    "GFCF": "#F58518",          # Orange
    "Inventories": "#72B7B2",   # Teal
    "Exports": "#54A24B",       # Green
    "Imports": "#E45756",       # Red
}

# Colors for current account components
CA_COLORS = {
    "Goods": "#54A24B",           # Green
    "Services": "#E45756",        # Red
    "Primary Income": "#4C78A8",  # Blue
    "Secondary Income": "#B279A2", # Purple
}

# Trade flow components (BOP item codes)
TRADE_COMPONENTS = {
    1000: "Goods Credits",
    2000: "Goods Debits",
    4000: "Services Credits",
    6000: "Services Debits",
}

# Colors for trade flow components
TRADE_COLORS = {
    "Goods Credits": "#54A24B",    # Green
    "Goods Debits": "#E45756",     # Red
    "Services Credits": "#4C78A8", # Blue
    "Services Debits": "#F58518",  # Orange
    "Goods Balance": "#72B7B2",    # Teal
    "Services Balance": "#B279A2", # Purple
}

# Colors for labour force measures
LF_COLORS = {
    "Unemployment Rate": "#E45756",      # Red
    "Participation Rate": "#4C78A8",     # Blue
    "Employment/Pop Ratio": "#54A24B",   # Green
}

# Shorten verbose ABS country names
COUNTRY_NAME_OVERRIDES = {
    "China (excludes SARs and Taiwan)": "China",
    "Korea, Republic of": "South Korea",
    "Taiwan, Province of China": "Taiwan",
    "Hong Kong (SAR of China)": "Hong Kong",
    "Macau (SAR of China)": "Macau",
    "United States of America": "United States",
    "United Kingdom, Channel Islands and Isle of Man": "United Kingdom",
    "Papua New Guinea": "PNG",
    "United Arab Emirates": "UAE",
    "Saudi Arabia": "Saudi Arabia",
    "New Zealand": "New Zealand",
}


def fetch_abs_csv(dataflow: str, query: str, start_period: str = "2015-Q1") -> pd.DataFrame:
    """Fetch data from ABS API in CSV format."""
    url = f"{ABS_API_BASE}/{dataflow}/{query}"
    params = {
        "startPeriod": start_period,
        "format": "csv",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def fetch_abs_codelist(codelist_id: str) -> Dict[str, str]:
    """Fetch an ABS SDMX codelist and return {code: name} mapping."""
    url = f"https://data.api.abs.gov.au/rest/codelist/ABS/{codelist_id}"
    response = requests.get(url, headers={"Accept": "application/xml"})
    response.raise_for_status()

    root = ET.fromstring(response.content)
    # SDMX namespace handling
    ns = {
        "mes": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
        "str": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
        "com": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
    }

    codes = {}
    for code_el in root.findall(".//str:Code", ns):
        code_id = code_el.attrib.get("id", "")
        name_el = code_el.find("com:Name", ns)
        if name_el is not None and name_el.text:
            codes[code_id] = name_el.text
    return codes


def get_gdp_contributions(start_period: str = "2015-Q1") -> pd.DataFrame:
    """Fetch and process GDP contribution data."""
    items = list(GDP_COMPONENTS.keys())
    items_str = "+".join(items)
    df = fetch_abs_csv("ABS,ANA_EXP", f"TCH.{items_str}.SSS.20.AUS.Q", start_period)

    # Filter and pivot
    df = df[df["DATA_ITEM"].isin(GDP_COMPONENTS.keys())].copy()
    df["component"] = df["DATA_ITEM"].map(GDP_COMPONENTS)

    # Pivot to wide format
    pivot = df.pivot_table(
        index="TIME_PERIOD",
        columns="component",
        values="OBS_VALUE",
        aggfunc="first"
    ).reset_index()

    # Sort by time period
    pivot = pivot.sort_values("TIME_PERIOD").reset_index(drop=True)

    return pivot


def get_gdp_growth(start_period: str = "2015-Q1") -> pd.DataFrame:
    """Fetch GDP growth rate."""
    df = fetch_abs_csv("ABS,ANA_EXP", "PCT_VCH.GPM.SSS.20.AUS.Q", start_period)
    df = df[df["DATA_ITEM"] == "GPM"][["TIME_PERIOD", "OBS_VALUE"]].copy()
    df = df.rename(columns={"OBS_VALUE": "GDP_growth"})
    df = df.sort_values("TIME_PERIOD").reset_index(drop=True)
    return df


def get_current_account(start_period: str = "2015-Q1") -> pd.DataFrame:
    """Fetch current account balance data."""
    # Items: 100=CA total, 170=Goods, 180=Services, 8700=Primary, 8100=Secondary
    items = [100] + list(CA_COMPONENTS.keys())
    items_str = "+".join(str(i) for i in items)
    df = fetch_abs_csv("ABS,BOP", f"1.{items_str}.20.Q", start_period)

    # Pivot to wide format
    pivot = df.pivot_table(
        index="TIME_PERIOD",
        columns="DATA_ITEM",
        values="OBS_VALUE",
        aggfunc="first"
    ).reset_index()

    # Convert from millions to billions
    for col in pivot.columns:
        if col != "TIME_PERIOD":
            pivot[col] = pivot[col] / 1000

    # Rename columns
    pivot = pivot.rename(columns={
        100: "Current Account",
        **CA_COMPONENTS
    })

    # Sort by time period
    pivot = pivot.sort_values("TIME_PERIOD").reset_index(drop=True)

    return pivot


def get_trade_data(start_period: str = "2015-Q1") -> pd.DataFrame:
    """Fetch trade flow data (goods/services exports and imports)."""
    items = list(TRADE_COMPONENTS.keys())
    items_str = "+".join(str(i) for i in items)
    df = fetch_abs_csv("ABS,BOP", f"1.{items_str}.20.Q", start_period)

    # Pivot to wide format
    pivot = df.pivot_table(
        index="TIME_PERIOD",
        columns="DATA_ITEM",
        values="OBS_VALUE",
        aggfunc="first"
    ).reset_index()

    # Convert from millions to billions
    for col in pivot.columns:
        if col != "TIME_PERIOD":
            pivot[col] = pivot[col] / 1000

    # Rename columns
    pivot = pivot.rename(columns=TRADE_COMPONENTS)

    # Debits are already negative from the BOP data; keep them negative

    # Compute balances
    if "Goods Credits" in pivot.columns and "Goods Debits" in pivot.columns:
        pivot["Goods Balance"] = pivot["Goods Credits"] + pivot["Goods Debits"]
    if "Services Credits" in pivot.columns and "Services Debits" in pivot.columns:
        pivot["Services Balance"] = pivot["Services Credits"] + pivot["Services Debits"]

    # Sort by time period
    pivot = pivot.sort_values("TIME_PERIOD").reset_index(drop=True)

    return pivot


def get_inflation_rba(start_year: int = 2015) -> pd.DataFrame:
    """Fetch inflation data from RBA G1 table (includes trimmed mean)."""
    from io import BytesIO

    url = "https://www.rba.gov.au/statistics/tables/xls/g01hist.xlsx"
    response = requests.get(url)
    response.raise_for_status()

    # Read Excel directly from response
    df = pd.read_excel(BytesIO(response.content), sheet_name="Data", header=None)

    # Data structure: Row 1 = titles, Row 11+ = data
    # Col 0 = dates, Col 1 = headline CPI index, Col 2 = year-ended headline
    # Col 10 = year-ended trimmed mean

    dates = pd.to_datetime(df.iloc[11:, 0], errors='coerce')
    headline = pd.to_numeric(df.iloc[11:, 2], errors='coerce')  # Year-ended inflation
    trimmed = pd.to_numeric(df.iloc[11:, 10], errors='coerce')  # Trimmed mean

    result = pd.DataFrame({
        'Date': dates.values,
        'Headline': headline.values,
        'Trimmed Mean': trimmed.values
    })

    # Filter and clean
    result = result.dropna(subset=['Date'])
    result = result[result['Date'].dt.year >= start_year]
    result = result.sort_values('Date').reset_index(drop=True)

    # Create quarterly period string for consistency with other charts (YYYY-Q#)
    result['TIME_PERIOD'] = result['Date'].dt.year.astype(str) + '-Q' + result['Date'].dt.quarter.astype(str)

    return result


def get_labour_force(start_period: str = "2015-01") -> pd.DataFrame:
    """Fetch labour force data (monthly, seasonally adjusted)."""
    # SEX=3 (Persons), AGE=1599 (15+), TSEST=20 (Seasonally Adjusted), REGION=AUS
    measures = list(LF_MEASURES.keys())
    measures_str = "+".join(measures)
    df = fetch_abs_csv("ABS,LF", f"{measures_str}.3.1599.20.AUS.M", start_period)

    # Pivot to wide format
    pivot = df.pivot_table(
        index="TIME_PERIOD",
        columns="MEASURE",
        values="OBS_VALUE",
        aggfunc="first"
    ).reset_index()

    # Rename columns
    pivot = pivot.rename(columns=LF_MEASURES)

    # Sort by time period and format as "Mon YYYY" for readability
    pivot = pivot.sort_values("TIME_PERIOD").reset_index(drop=True)

    # Convert TIME_PERIOD from "2024-08" to "Aug 2024" format
    pivot['TIME_PERIOD'] = pd.to_datetime(pivot['TIME_PERIOD']).dt.strftime('%b %Y')

    return pivot


def _process_merch_table(
    df: pd.DataFrame,
    group_col: str,
    name_lookup: Dict[str, str],
) -> pd.DataFrame:
    """Process a raw MERCH_EXP / MERCH_IMP CSV into a summary table.

    Returns a DataFrame with columns:
        Name, latest_qtr, trailing_4q, qoq_pct, yoy_pct, share_pct
    sorted by trailing_4q descending, top 20 rows.
    """
    # group_col is either 'SITC' or 'COUNTRY' depending on the breakdown
    df = df.copy()
    # Ensure group column is string so it matches codelist keys (e.g. 1 -> "001")
    df[group_col] = df[group_col].astype(str).str.zfill(3) if "SITC" in group_col else df[group_col].astype(str)
    # Drop total rows (code == "TOT" or starts with "_T")
    df = df[~df[group_col].isin(["TOT", "_T", "_Z"])].copy()
    # Convert AUD thousands → billions
    df["value_bn"] = df["OBS_VALUE"] / 1_000_000

    # Parse TIME_PERIOD (monthly: "2024-07") into datetime
    df["date"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m")
    df["quarter"] = df["date"].dt.to_period("Q")

    # Aggregate monthly → quarterly
    qtr = df.groupby([group_col, "quarter"])["value_bn"].agg(["sum", "count"]).reset_index()
    qtr.columns = [group_col, "quarter", "value", "n_months"]

    # Drop incomplete latest quarter (< 3 months)
    max_q = qtr["quarter"].max()
    if qtr.loc[qtr["quarter"] == max_q, "n_months"].max() < 3:
        qtr = qtr[qtr["quarter"] != max_q]

    if qtr.empty:
        return pd.DataFrame()

    # Pivot: rows=quarter, cols=code
    piv = qtr.pivot_table(index="quarter", columns=group_col, values="value", aggfunc="sum")
    piv = piv.sort_index()

    if len(piv) < 2:
        return pd.DataFrame()

    latest_q = piv.index[-1]
    latest_vals = piv.iloc[-1]

    # Trailing 4 quarters
    n_trail = min(4, len(piv))
    trailing_4q = piv.iloc[-n_trail:].sum()

    # Q-o-Q %
    prev_q_vals = piv.iloc[-2]
    qoq_pct = ((latest_vals / prev_q_vals) - 1) * 100

    # YoY % (trailing 4Q vs prior 4Q)
    if len(piv) >= 2 * n_trail:
        prior_4q = piv.iloc[-2 * n_trail:-n_trail].sum()
        yoy_pct = ((trailing_4q / prior_4q) - 1) * 100
    else:
        yoy_pct = pd.Series(np.nan, index=piv.columns)

    # Share %
    total_trailing = trailing_4q.sum()
    share_pct = (trailing_4q / total_trailing) * 100 if total_trailing != 0 else trailing_4q * 0

    # Build result
    records = []
    for code in piv.columns:
        name = name_lookup.get(code, code)
        # Tidy SITC commodity names: strip parenthetical qualifiers and cap length
        if "SITC" in group_col and name != code:
            import re
            # Remove parenthetical content
            name = re.sub(r'\s*\(.*?\)', '', name).strip()
            # Remove trailing ", nes" / ", not elsewhere specified"
            name = re.sub(r',?\s*n\.?e\.?s\.?$', '', name, flags=re.IGNORECASE).strip()
            name = re.sub(r',?\s*not elsewhere specified$', '', name, flags=re.IGNORECASE).strip()
            # Remove trailing ", whether or not ..."
            name = re.sub(r',?\s*whether or not.*$', '', name, flags=re.IGNORECASE).strip()
            # Remove trailing comma
            name = name.rstrip(',').strip()
            # Capitalize first letter
            if name:
                name = name[0].upper() + name[1:]
        # Apply country overrides
        name = COUNTRY_NAME_OVERRIDES.get(name, name)
        records.append({
            "Name": name,
            "latest_qtr": latest_vals.get(code, np.nan),
            "trailing_4q": trailing_4q.get(code, np.nan),
            "qoq_pct": qoq_pct.get(code, np.nan),
            "yoy_pct": yoy_pct.get(code, np.nan),
            "share_pct": share_pct.get(code, np.nan),
            "_latest_q_label": str(latest_q),
        })

    result = pd.DataFrame(records)
    result = result.sort_values("trailing_4q", ascending=False).head(20).reset_index(drop=True)
    return result


# Regional aggregates to exclude from services country tables
_SERVICES_COUNTRY_AGGREGATES = {
    "TOT", "APEC", "OECD", "ASEA", "EU27", "EURO", "OPEC",
    "OTHE", "BRIC", "G20", "EASIA", "SASIA", "AMER", "EURP",
    "AFRI", "OCEA", "MIDE", "REST", "WOTHR",
}


def _process_services_table(
    df: pd.DataFrame,
    group_col: str,
    name_lookup: Dict[str, str],
    abs_values: bool = True,
) -> pd.DataFrame:
    """Process a raw services trade CSV into a summary table.

    Returns a DataFrame with columns:
        Name, latest_yr, prior_yr, yoy_pct, share_pct, _latest_yr_label
    sorted by latest_yr descending, top 15 rows.
    """
    df = df.copy()
    df[group_col] = df[group_col].astype(str)

    # Filter out totals and aggregates
    if group_col == "EBOPS":
        # Keep only top-level EBOPS codes (1-digit or 2-digit 1-12)
        def _is_top_level(code):
            if code in ("TOTAL", "TOT", "_T", "_Z"):
                return False
            try:
                n = int(code)
                return 1 <= n <= 12
            except ValueError:
                return False
        df = df[df[group_col].apply(_is_top_level)]
    else:
        # Country: exclude TOT and regional aggregates
        df = df[~df[group_col].isin(_SERVICES_COUNTRY_AGGREGATES)]

    # Convert AUD millions → billions
    df["value_bn"] = df["OBS_VALUE"] / 1000
    if abs_values:
        df["value_bn"] = df["value_bn"].abs()

    # Pivot: rows = TIME_PERIOD (year), columns = code
    piv = df.pivot_table(index="TIME_PERIOD", columns=group_col, values="value_bn", aggfunc="sum")
    piv = piv.sort_index()

    if len(piv) < 2:
        return pd.DataFrame()

    latest_yr = piv.index[-1]
    prior_yr = piv.index[-2]
    latest_vals = piv.loc[latest_yr]
    prior_vals = piv.loc[prior_yr]

    yoy_pct = ((latest_vals / prior_vals) - 1) * 100

    total_latest = latest_vals.sum()
    share_pct = (latest_vals / total_latest) * 100 if total_latest != 0 else latest_vals * 0

    # Determine year label (detect FY vs CY from TIME_PERIOD format)
    yr_str = str(latest_yr)
    yr_label = f"FY{yr_str}" if "-" in yr_str else yr_str

    records = []
    for code in piv.columns:
        name = name_lookup.get(code, code)
        name = COUNTRY_NAME_OVERRIDES.get(name, name)
        records.append({
            "Name": name,
            "latest_yr": latest_vals.get(code, np.nan),
            "prior_yr": prior_vals.get(code, np.nan),
            "yoy_pct": yoy_pct.get(code, np.nan),
            "share_pct": share_pct.get(code, np.nan),
            "_latest_yr_label": yr_label,
        })

    result = pd.DataFrame(records)
    result = result.sort_values("latest_yr", ascending=False).head(15).reset_index(drop=True)
    return result


def get_merch_trade_tables(start_period: str = "2023-01") -> Dict[str, pd.DataFrame]:
    """Fetch merchandise trade data and return 4 summary DataFrames.

    Keys: 'exp_commodity', 'imp_commodity', 'exp_country', 'imp_country'
    """
    print("Fetching merchandise trade codelists...")
    sitc_names = fetch_abs_codelist("CL_MERCH_SITC")
    country_names = fetch_abs_codelist("CL_MERCH_COUNTRY")

    # Use 3-digit SITC codes derived from the codelist
    sitc_3digit = sorted(c for c in sitc_names if len(c) == 3 and c.isdigit())
    sitc_filter = "+".join(sitc_3digit)

    # Dimension order: COMMODITY_SITC.COUNTRY.STATE.FREQ
    print("Fetching merchandise exports by commodity...")
    exp_comm = fetch_abs_csv(
        "ABS,MERCH_EXP",
        f"{sitc_filter}.TOT.TOT.M",
        start_period,
    )

    print("Fetching merchandise imports by commodity...")
    imp_comm = fetch_abs_csv(
        "ABS,MERCH_IMP",
        f"{sitc_filter}.TOT.TOT.M",
        start_period,
    )

    print("Fetching merchandise exports by country...")
    exp_ctry = fetch_abs_csv(
        "ABS,MERCH_EXP",
        f"TOT..TOT.M",
        start_period,
    )

    print("Fetching merchandise imports by country...")
    imp_ctry = fetch_abs_csv(
        "ABS,MERCH_IMP",
        f"TOT..TOT.M",
        start_period,
    )

    tables = {}
    tables["exp_commodity"] = _process_merch_table(exp_comm, "COMMODITY_SITC", sitc_names)
    tables["imp_commodity"] = _process_merch_table(imp_comm, "COMMODITY_SITC", sitc_names)
    tables["exp_country"] = _process_merch_table(exp_ctry, "COUNTRY_DEST", country_names)
    tables["imp_country"] = _process_merch_table(imp_ctry, "COUNTRY_ORIGIN", country_names)

    return tables


def get_services_trade_tables() -> Dict[str, pd.DataFrame]:
    """Fetch services trade data and return 4 summary DataFrames.

    Keys: 'svc_exp_type', 'svc_imp_type', 'svc_exp_country', 'svc_imp_country'
    """
    import datetime

    print("Fetching services trade codelists...")
    ebops_names = fetch_abs_codelist("CL_EBOPS_TRADE")
    country_names = fetch_abs_codelist("CL_SERVICES_COUNTRY")

    # Determine which dataflow (CY or FY) has the most recent data
    print("Checking services trade data availability...")
    cy_flow = "ABS,TRADE_SERV_CNTRY_CY"
    fy_flow = "ABS,TRADE_SERV_CNTRY_FY"

    try:
        cy_total = fetch_abs_csv(cy_flow, "EXP.TOTAL.TOT.A", "2015")
        cy_max_yr = cy_total["TIME_PERIOD"].max()
    except Exception:
        cy_max_yr = "0"

    try:
        fy_total = fetch_abs_csv(fy_flow, "EXP.TOTAL.TOT.A", "2015")
        fy_max_yr = fy_total["TIME_PERIOD"].max()
    except Exception:
        fy_max_yr = "0"

    if str(fy_max_yr) > str(cy_max_yr):
        dataflow = fy_flow
        print(f"Using FY dataflow (latest: {fy_max_yr})")
    else:
        dataflow = cy_flow
        print(f"Using CY dataflow (latest: {cy_max_yr})")

    # Start period ~5 years back
    current_year = datetime.date.today().year
    start_yr = str(current_year - 5)

    print("Fetching services exports by type...")
    exp_type = fetch_abs_csv(dataflow, "EXP..TOT.A", start_yr)

    print("Fetching services imports by type...")
    imp_type = fetch_abs_csv(dataflow, "IMP..TOT.A", start_yr)

    print("Fetching services exports by country...")
    exp_ctry = fetch_abs_csv(dataflow, "EXP.TOTAL..A", start_yr)

    print("Fetching services imports by country...")
    imp_ctry = fetch_abs_csv(dataflow, "IMP.TOTAL..A", start_yr)

    tables = {}
    tables["svc_exp_type"] = _process_services_table(exp_type, "EBOPS", ebops_names)
    tables["svc_imp_type"] = _process_services_table(imp_type, "EBOPS", ebops_names, abs_values=True)
    tables["svc_exp_country"] = _process_services_table(exp_ctry, "COUNTRY", country_names)
    tables["svc_imp_country"] = _process_services_table(imp_ctry, "COUNTRY", country_names, abs_values=True)

    return tables


# =============================================================================
# Automatic Reporting / Anomaly Detection
# =============================================================================

def detect_outlier(series: pd.Series, threshold: float = 2.0) -> Tuple[bool, float, float]:
    """Check if the latest value is an outlier (>threshold std from mean)."""
    if len(series) < 5:
        return False, 0, 0
    historical = series.iloc[:-1]
    latest = series.iloc[-1]
    mean = historical.mean()
    std = historical.std()
    if std == 0:
        return False, 0, 0
    z_score = (latest - mean) / std
    return abs(z_score) > threshold, z_score, latest


def detect_large_change(series: pd.Series, threshold: float = 2.0) -> Tuple[bool, float, float]:
    """Check if latest period-on-period change is unusually large."""
    if len(series) < 5:
        return False, 0, 0
    changes = series.diff().dropna()
    if len(changes) < 4:
        return False, 0, 0
    historical_changes = changes.iloc[:-1]
    latest_change = changes.iloc[-1]
    mean = historical_changes.mean()
    std = historical_changes.std()
    if std == 0:
        return False, 0, 0
    z_score = (latest_change - mean) / std
    return abs(z_score) > threshold, z_score, latest_change


def detect_consecutive_trend(series: pd.Series, min_periods: int = 3) -> Tuple[bool, int, str]:
    """Detect if series has moved in same direction for min_periods consecutive periods."""
    if len(series) < min_periods + 1:
        return False, 0, ""
    changes = series.diff().dropna()

    # Count consecutive movements at the end
    direction = None
    count = 0
    for change in reversed(changes.values):
        if direction is None:
            direction = "up" if change > 0 else "down"
            count = 1
        elif (direction == "up" and change > 0) or (direction == "down" and change < 0):
            count += 1
        else:
            break

    return count >= min_periods, count, direction if count >= min_periods else ""


def detect_cumulative_trend(series: pd.Series, window: int = 4, threshold: float = 1.5) -> Tuple[bool, float, float]:
    """
    Check if cumulative movement over last N periods is unusual compared to
    historical N-period movements.
    """
    if len(series) < window + 8:  # Need enough history
        return False, 0, 0

    # Calculate all rolling cumulative changes
    cumulative_changes = series.diff(window).dropna()
    if len(cumulative_changes) < 4:
        return False, 0, 0

    historical = cumulative_changes.iloc[:-1]
    latest = cumulative_changes.iloc[-1]
    mean = historical.mean()
    std = historical.std()
    if std == 0:
        return False, 0, 0
    z_score = (latest - mean) / std
    return abs(z_score) > threshold, z_score, latest


def detect_threshold_crossing(
    series: pd.Series,
    lower: float = None,
    upper: float = None,
    lookback: int = 4
) -> Tuple[bool, str, float]:
    """Check if series has crossed a threshold recently."""
    if len(series) < lookback + 1:
        return False, "", 0

    recent = series.iloc[-lookback:]
    earlier = series.iloc[-lookback - 1]
    latest = series.iloc[-1]

    if upper is not None:
        # Check if crossed above upper threshold
        if earlier <= upper and latest > upper:
            return True, "above", latest
        # Check if crossed below upper threshold (returning to range)
        if earlier > upper and latest <= upper:
            return True, "returned below", latest

    if lower is not None:
        # Check if crossed below lower threshold
        if earlier >= lower and latest < lower:
            return True, "below", latest
        # Check if crossed above lower threshold (returning to range)
        if earlier < lower and latest >= lower:
            return True, "returned above", latest

    return False, "", 0


def detect_rolling_trend(
    series: pd.Series,
    window: int = 8,
    min_slope_threshold: float = 0.15
) -> Tuple[bool, str, bool]:
    """Detect trend direction via OLS over a trailing window.

    Returns (has_trend, slope_description, is_flat).
    slope_description is one of: "trending up", "trending down",
    "decelerating", "accelerating", or "" if no signal.
    """
    s = series.dropna()
    if len(s) < window:
        return False, "", False

    y = s.iloc[-window:].values
    x = np.arange(len(y), dtype=float)
    std = np.std(y)
    if std == 0:
        return False, "", True

    # Full-window slope (normalised per-period change as fraction of std)
    coeffs = np.polyfit(x, y, 1)
    slope_full = coeffs[0]
    norm_slope = slope_full / std

    # Compute R² for flatness check
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Slope significance: t = slope / se(slope)
    n = len(y)
    mse = ss_res / (n - 2) if n > 2 else ss_res
    ss_xx = np.sum((x - np.mean(x)) ** 2)
    se_slope = np.sqrt(mse / ss_xx) if ss_xx > 0 else np.inf
    t_slope = slope_full / se_slope if se_slope > 0 else 0.0

    # Flat detection
    if abs(norm_slope) < min_slope_threshold and r_sq < 0.3:
        return True, "", True

    # Require both a meaningful effect size AND statistical significance
    if abs(norm_slope) < min_slope_threshold or abs(t_slope) < 1.645:
        return False, "", False

    # Half-window comparison for acceleration/deceleration
    half = window // 2
    if half >= 3:
        y_recent = s.iloc[-half:].values
        x_recent = np.arange(len(y_recent), dtype=float)
        slope_recent = np.polyfit(x_recent, y_recent, 1)[0]

        # Sign flip → deceleration/acceleration
        if slope_full > 0 and slope_recent < -slope_full * 0.3:
            return True, "decelerating", False
        if slope_full < 0 and slope_recent > -slope_full * 0.3:
            return True, "decelerating", False
        # Magnitude divergence (recent much stronger)
        if abs(slope_recent) > abs(slope_full) * 2:
            label = "accelerating" if slope_recent * slope_full > 0 else "decelerating"
            return True, label, False

    direction = "trending up" if slope_full > 0 else "trending down"
    return True, direction, False


def detect_volatility_change(
    series: pd.Series,
    recent_window: int = 6,
    historical_window: int = 6
) -> Tuple[bool, str, float]:
    """Detect if recent volatility differs from historical.

    Returns (has_change, direction, ratio) where direction is
    "more volatile" or "stabilised".
    """
    s = series.dropna()
    total_needed = recent_window + historical_window
    if len(s) < total_needed:
        return False, "", 1.0

    recent = s.iloc[-recent_window:]
    historical = s.iloc[-(total_needed):-recent_window]

    recent_std = recent.std()
    hist_std = historical.std()

    if hist_std == 0:
        return False, "", 1.0

    ratio = recent_std / hist_std

    if ratio > 1.8:
        return True, "more volatile", ratio
    elif ratio < 0.5:
        return True, "stabilised", ratio
    return False, "", ratio


def detect_base_effect(
    qoq_series: pd.Series,
    window: int = 4
) -> Tuple[bool, float, float]:
    """Detect base-effect divergence between YoY and annualised QoQ.

    qoq_series should be quarter-on-quarter changes.
    Returns (has_base_effect, yoy_approx, qoq_annualised).
    """
    s = qoq_series.dropna()
    if len(s) < window + 1:
        return False, 0.0, 0.0

    # Approximate YoY as rolling sum of last `window` QoQ changes
    yoy_approx = s.iloc[-window:].sum()
    qoq_ann = s.iloc[-1] * window

    divergence = abs(yoy_approx - qoq_ann)
    threshold = max(0.8, abs(yoy_approx) * 0.4)

    # Check if the quarter dropping out is extreme
    dropping_out = s.iloc[-(window + 1)]
    median_change = s.iloc[-window:].median()
    extreme_dropout = abs(dropping_out - median_change) > abs(median_change) * 1.5

    if divergence > threshold and extreme_dropout:
        return True, yoy_approx, qoq_ann
    return False, yoy_approx, qoq_ann


def detect_persistence_change(
    series: pd.Series,
    window: int = 8,
    baseline_window: int = 8
) -> Tuple[bool, str, float, float]:
    """Detect change in AR(1) persistence.

    Returns (has_change, direction, recent_ar1, baseline_ar1).
    direction is "more persistent" or "less persistent".
    """
    s = series.dropna()
    total_needed = window + baseline_window
    if len(s) < total_needed + 1:
        return False, "", 0.0, 0.0

    recent = s.iloc[-window:]
    baseline = s.iloc[-(total_needed):-window]

    # AR(1) = correlation between y_t and y_{t-1}
    def _ar1(segment):
        if len(segment) < 4:
            return 0.0
        y = segment.values
        return np.corrcoef(y[1:], y[:-1])[0, 1]

    recent_ar1 = _ar1(recent)
    baseline_ar1 = _ar1(baseline)

    if np.isnan(recent_ar1) or np.isnan(baseline_ar1):
        return False, "", 0.0, 0.0

    diff = recent_ar1 - baseline_ar1
    if abs(diff) > 0.3:
        direction = "more persistent" if diff > 0 else "less persistent"
        return True, direction, recent_ar1, baseline_ar1
    return False, "", recent_ar1, baseline_ar1


def find_trend_window(
    series: pd.Series,
    min_window: int = 4,
    max_window: int = 40
) -> int:
    """Find the trend window via lightweight structural break detection.

    Scans from the most recent candidate break point outward, returning the
    distance from the first significant mean shift to the end of the series.
    Falls back to max_window when no break is found.
    """
    s = series.dropna().values
    n = len(s)
    # Cap so there is data before the break to compare against
    max_window = min(max_window, n - 3)
    if max_window < min_window:
        return min(min_window, n)

    overall_std = np.std(s)
    if overall_std == 0:
        return max_window

    for k in range(min_window, max_window + 1):
        post = s[-k:]
        # Need an equal-length segment before the break
        if k > n - k:
            continue
        pre = s[-2 * k:-k]

        mean_post = np.mean(post)
        mean_pre = np.mean(pre)
        mean_diff = mean_post - mean_pre

        # Effect size gate: difference must be meaningful relative to overall std
        if abs(mean_diff) < 0.5 * overall_std:
            continue

        # Welch t-statistic
        var_post = np.var(post, ddof=1)
        var_pre = np.var(pre, ddof=1)
        denom = np.sqrt(var_post / len(post) + var_pre / len(pre))
        if denom == 0:
            continue
        t_stat = mean_diff / denom

        if abs(t_stat) > 2.5:
            return k

    return max_window


def generate_insights(
    gdp_df: pd.DataFrame,
    ca_data: pd.DataFrame,
    inflation_data: pd.DataFrame,
    lf_data: pd.DataFrame
) -> dict:
    """Generate bullet-point insights from the data, grouped by category."""
    insights: dict[str, list[str]] = {
        "GDP": [],
        "Current Account": [],
        "Inflation": [],
        "Labour Market": [],
    }

    # Map series names used in narrative tests to insight categories
    _category_map = {
        "GDP growth": "GDP",
        "GDP": "GDP",
        "CA balance": "Current Account",
        "Trimmed mean inflation": "Inflation",
        "Inflation": "Inflation",
        "Unemployment": "Labour Market",
        "Participation rate": "Labour Market",
    }

    # Get the latest period for context
    latest_gdp_period = gdp_df["TIME_PERIOD"].iloc[-1] if len(gdp_df) > 0 else "recent"
    latest_ca_period = ca_data["TIME_PERIOD"].iloc[-1] if len(ca_data) > 0 else "recent"
    latest_infl_period = inflation_data["TIME_PERIOD"].iloc[-1] if len(inflation_data) > 0 else "recent"
    latest_lf_period = lf_data["TIME_PERIOD"].iloc[-1] if len(lf_data) > 0 else "recent"

    # ----- GDP Analysis -----
    if "GDP_growth" in gdp_df.columns:
        gdp_series = gdp_df["GDP_growth"].dropna()

        # Check for outlier GDP growth
        is_outlier, z, val = detect_outlier(gdp_series, threshold=1.5)
        if is_outlier:
            direction = "strong" if z > 0 else "weak"
            insights["GDP"].append(f"GDP growth was unusually {direction} in {latest_gdp_period} at {val:.1f}%")

        # Check for consecutive trend
        is_trend, count, direction = detect_consecutive_trend(gdp_series, min_periods=3)
        if is_trend:
            trend_word = "risen" if direction == "up" else "fallen"
            insights["GDP"].append(f"GDP growth has {trend_word} for {count} consecutive quarters")

        # Check cumulative trend over 4 and 8 quarters
        for window, period_desc in [(4, "past year"), (8, "past 2 years")]:
            is_cum_trend, z, cum_change = detect_cumulative_trend(gdp_series, window=window, threshold=1.3)
            if is_cum_trend and abs(cum_change) > 0.3:
                direction = "accelerated" if cum_change > 0 else "slowed"
                insights["GDP"].append(f"GDP growth has {direction} over the {period_desc} (cumulative change: {cum_change:+.1f} ppts)")
                break  # Only report one

    # Check GDP components for large swings
    for comp in ["Consumption", "GFCF", "Inventories", "Exports", "Imports"]:
        if comp in gdp_df.columns:
            series = gdp_df[comp].dropna()
            is_large, z, change = detect_large_change(series, threshold=1.8)
            if is_large:
                direction = "jumped" if change > 0 else "dropped"
                insights["GDP"].append(f"{comp} contribution {direction} by {abs(change):.1f} ppts in {latest_gdp_period}")

            # Check cumulative trend over 4 quarters
            is_cum_trend, z, cum_change = detect_cumulative_trend(series, window=4, threshold=1.3)
            if is_cum_trend and abs(cum_change) > 0.3:
                direction = "strengthened" if cum_change > 0 else "weakened"
                insights["GDP"].append(f"{comp} contributions have {direction} over the past year ({cum_change:+.1f} ppts)")

    # ----- Current Account Analysis -----
    if "Current Account" in ca_data.columns:
        ca_series = ca_data["Current Account"].dropna()

        # Check for outlier
        is_outlier, z, val = detect_outlier(ca_series, threshold=1.5)
        if is_outlier:
            direction = "surplus" if val > 0 else "deficit"
            size = "large" if abs(z) > 2 else "notable"
            insights["Current Account"].append(f"Current account recorded a {size} {direction} of ${abs(val):.1f}bn in {latest_ca_period}")

        # Check for cumulative trend (4 and 8 quarters)
        for window, period_desc in [(4, "past year"), (8, "past 2 years")]:
            is_cum_trend, z, cum_change = detect_cumulative_trend(ca_series, window=window, threshold=1.3)
            if is_cum_trend and abs(cum_change) > 3:
                direction = "improved" if cum_change > 0 else "deteriorated"
                insights["Current Account"].append(f"Current account has {direction} by ${abs(cum_change):.1f}bn over the {period_desc}")
                break

    # Check CA components
    for comp in ["Goods", "Services", "Primary Income"]:
        if comp in ca_data.columns:
            series = ca_data[comp].dropna()
            is_cum_trend, z, cum_change = detect_cumulative_trend(series, window=4, threshold=1.3)
            if is_cum_trend and abs(cum_change) > 1.5:
                direction = "improved" if cum_change > 0 else "worsened"
                insights["Current Account"].append(f"{comp} balance has {direction} by ${abs(cum_change):.1f}bn over the past year")

    # ----- Inflation Analysis -----
    if "Trimmed Mean" in inflation_data.columns:
        infl_series = inflation_data["Trimmed Mean"].dropna()
        latest_infl = infl_series.iloc[-1] if len(infl_series) > 0 else None

        if latest_infl is not None:
            # Check if outside RBA target band
            if latest_infl > 3:
                insights["Inflation"].append(f"Trimmed mean inflation ({latest_infl:.1f}%) remains above the RBA's 2-3% target band")
            elif latest_infl < 2:
                insights["Inflation"].append(f"Trimmed mean inflation ({latest_infl:.1f}%) is below the RBA's 2-3% target band")

            # Check for threshold crossing
            is_cross, cross_type, val = detect_threshold_crossing(infl_series, lower=2, upper=3, lookback=2)
            if is_cross:
                if "returned" in cross_type:
                    insights["Inflation"].append(f"Inflation has {cross_type} the RBA target band at {val:.1f}%")

        # Check cumulative trend (4 and 8 quarters)
        for window, period_desc in [(4, "past year"), (8, "past 2 years")]:
            is_cum_trend, z, cum_change = detect_cumulative_trend(infl_series, window=window, threshold=1.3)
            if is_cum_trend and abs(cum_change) > 0.3:
                direction = "risen" if cum_change > 0 else "fallen"
                insights["Inflation"].append(f"Trimmed mean inflation has {direction} by {abs(cum_change):.1f} ppts over the {period_desc}")
                break

    # ----- Labour Market Analysis -----
    # Need to handle monthly data - use 12 months as "year"
    if "Unemployment Rate" in lf_data.columns:
        unemp_series = lf_data["Unemployment Rate"].dropna()

        # Check for outlier
        is_outlier, z, val = detect_outlier(unemp_series, threshold=1.5)
        if is_outlier:
            level = "low" if z < 0 else "elevated"
            insights["Labour Market"].append(f"Unemployment rate is at a historically {level} level of {val:.1f}%")

        # Check cumulative trend (6 and 12 months)
        for window, period_desc in [(6, "past 6 months"), (12, "past year")]:
            is_cum_trend, z, cum_change = detect_cumulative_trend(unemp_series, window=window, threshold=1.3)
            if is_cum_trend and abs(cum_change) > 0.2:
                direction = "risen" if cum_change > 0 else "fallen"
                insights["Labour Market"].append(f"Unemployment rate has {direction} by {abs(cum_change):.1f} ppts over the {period_desc}")
                break

    if "Participation Rate" in lf_data.columns:
        part_series = lf_data["Participation Rate"].dropna()

        # Check for outlier (record high/low)
        is_outlier, z, val = detect_outlier(part_series, threshold=1.8)
        if is_outlier:
            level = "high" if z > 0 else "low"
            insights["Labour Market"].append(f"Participation rate is at a historically {level} level of {val:.1f}%")

        # Check cumulative trend
        for window, period_desc in [(6, "past 6 months"), (12, "past year")]:
            is_cum_trend, z, cum_change = detect_cumulative_trend(part_series, window=window, threshold=1.3)
            if is_cum_trend and abs(cum_change) > 0.2:
                direction = "risen" if cum_change > 0 else "fallen"
                insights["Labour Market"].append(f"Participation rate has {direction} by {abs(cum_change):.1f} ppts over the {period_desc}")
                break

    # ----- Narrative Detection Tests -----
    # Track (series_name, direction_keyword) for deduplication against existing insights
    def _dedup_ok(new_insight: str, series_name: str, direction_kw: str) -> bool:
        """Return True if the new insight is not redundant with existing ones."""
        sn_lower = series_name.lower()
        dk_lower = direction_kw.lower()
        for category_items in insights.values():
            for existing in category_items:
                ex_lower = existing.lower()
                if sn_lower in ex_lower and dk_lower in ex_lower:
                    return False
        return True

    # -- Rolling trend --
    def _format_window(n: int, unit: str, at_cap: bool = False) -> str:
        """Format a window size into a readable time description."""
        prefix = "at least the " if at_cap else "the "
        if unit == "quarters":
            years, rem = divmod(n, 4)
            if rem == 0 and years >= 1:
                base = "past year" if years == 1 else f"past {years} years"
            else:
                base = f"past {n} quarters"
        else:  # months
            years, rem = divmod(n, 12)
            if rem == 0 and years >= 1:
                base = "past year" if years == 1 else f"past {years} years"
            else:
                base = f"past {n} months"
        return prefix + base

    trend_targets_q = {
        "GDP growth": gdp_df["GDP_growth"].dropna() if "GDP_growth" in gdp_df.columns else pd.Series(dtype=float),
        "CA balance": ca_data["Current Account"].dropna() if "Current Account" in ca_data.columns else pd.Series(dtype=float),
        "Trimmed mean inflation": inflation_data["Trimmed Mean"].dropna() if "Trimmed Mean" in inflation_data.columns else pd.Series(dtype=float),
    }
    trend_targets_m = {
        "Unemployment": lf_data["Unemployment Rate"].dropna() if "Unemployment Rate" in lf_data.columns else pd.Series(dtype=float),
        "Participation rate": lf_data["Participation Rate"].dropna() if "Participation Rate" in lf_data.columns else pd.Series(dtype=float),
    }

    for name, s in trend_targets_q.items():
        cat = _category_map[name]
        w = find_trend_window(s, min_window=4, max_window=40)
        if len(s) < w:
            continue
        window_desc = _format_window(w, "quarters", at_cap=(w >= 40))
        has_trend, desc, is_flat = detect_rolling_trend(s, window=w)
        if is_flat and _dedup_ok(name, name, "flat"):
            insights[cat].append(f"{name} has been essentially flat for {window_desc}")
        elif has_trend and desc:
            dir_kw = "up" if "up" in desc or "accelerat" in desc else "down"
            if _dedup_ok(name, name, dir_kw):
                if "decelerat" in desc:
                    insights[cat].append(f"{name} has been decelerating — the recent trend is weaker than {window_desc} trend")
                elif "accelerat" in desc:
                    insights[cat].append(f"{name} has been accelerating — recent momentum exceeds {window_desc} trend")
                else:
                    direction = "up" if "up" in desc else "down"
                    insights[cat].append(f"{name} has been trending {direction} over {window_desc}")

    for name, s in trend_targets_m.items():
        cat = _category_map[name]
        w = find_trend_window(s, min_window=12, max_window=120)
        if len(s) < w:
            continue
        window_desc = _format_window(w, "months", at_cap=(w >= 120))
        has_trend, desc, is_flat = detect_rolling_trend(s, window=w)
        if is_flat and _dedup_ok(name, name, "flat"):
            insights[cat].append(f"{name} has been essentially flat for {window_desc}")
        elif has_trend and desc:
            dir_kw = "up" if "up" in desc or "accelerat" in desc else "down"
            if _dedup_ok(name, name, dir_kw):
                if "decelerat" in desc:
                    insights[cat].append(f"{name} has been decelerating — the recent trend is weaker than {window_desc} trend")
                elif "accelerat" in desc:
                    insights[cat].append(f"{name} has been accelerating — recent momentum exceeds {window_desc} trend")
                else:
                    direction = "up" if "up" in desc else "down"
                    insights[cat].append(f"{name} has been trending {direction} over {window_desc}")

    # -- Volatility change --
    vol_targets_q = {
        "GDP growth": gdp_df["GDP_growth"].dropna() if "GDP_growth" in gdp_df.columns else pd.Series(dtype=float),
        "CA balance": ca_data["Current Account"].dropna() if "Current Account" in ca_data.columns else pd.Series(dtype=float),
        "Trimmed mean inflation": inflation_data["Trimmed Mean"].dropna() if "Trimmed Mean" in inflation_data.columns else pd.Series(dtype=float),
        "Unemployment": lf_data["Unemployment Rate"].dropna() if "Unemployment Rate" in lf_data.columns else pd.Series(dtype=float),
    }
    for name, s in vol_targets_q.items():
        cat = _category_map[name]
        # Use 6-period windows for quarterly, 12 for monthly (unemployment)
        rw = 12 if name == "Unemployment" else 6
        has_change, direction, ratio = detect_volatility_change(s, recent_window=rw, historical_window=rw)
        if has_change:
            if direction == "more volatile":
                insights[cat].append(f"{name} has become notably more volatile over the past {rw} {'months' if name == 'Unemployment' else 'quarters'}")
            else:
                insights[cat].append(f"{name} has stabilised — recent {'monthly' if name == 'Unemployment' else 'quarterly'} variation is well below its historical norm")

    # -- Base effect --
    base_targets = {
        "GDP": (gdp_df["GDP_growth"].dropna() if "GDP_growth" in gdp_df.columns else pd.Series(dtype=float)),
        "Inflation": (inflation_data["Trimmed Mean"].diff().dropna() if "Trimmed Mean" in inflation_data.columns else pd.Series(dtype=float)),
    }
    for name, s in base_targets.items():
        cat = _category_map[name]
        if len(s) < 5:
            continue
        has_be, yoy, qoq_ann = detect_base_effect(s, window=4)
        if has_be:
            if yoy > qoq_ann:
                insights[cat].append(f"{name} YoY growth appears strong but current quarterly momentum is weaker — partly a base effect from the weak quarter dropping out of the annual comparison")
            else:
                insights[cat].append(f"{name}'s year-ended rate overstates current momentum — recent quarterly changes have been more moderate")

    # -- Persistence change --
    persist_targets_q = {
        "GDP growth": gdp_df["GDP_growth"].dropna() if "GDP_growth" in gdp_df.columns else pd.Series(dtype=float),
        "CA balance": ca_data["Current Account"].dropna() if "Current Account" in ca_data.columns else pd.Series(dtype=float),
        "Trimmed mean inflation": inflation_data["Trimmed Mean"].dropna() if "Trimmed Mean" in inflation_data.columns else pd.Series(dtype=float),
    }
    persist_targets_m = {
        "Unemployment": lf_data["Unemployment Rate"].dropna() if "Unemployment Rate" in lf_data.columns else pd.Series(dtype=float),
    }

    for name, s in persist_targets_q.items():
        cat = _category_map[name]
        has_change, direction, r_ar1, b_ar1 = detect_persistence_change(s, window=8, baseline_window=8)
        if has_change:
            if "more" in direction:
                insights[cat].append(f"{name} movements have become more persistent — each change tends to stick rather than reverse")
            else:
                insights[cat].append(f"{name} has become less predictable from one quarter to the next")

    for name, s in persist_targets_m.items():
        cat = _category_map[name]
        has_change, direction, r_ar1, b_ar1 = detect_persistence_change(s, window=18, baseline_window=18)
        if has_change:
            if "more" in direction:
                insights[cat].append(f"{name} movements have become more persistent — each change tends to stick rather than reverse")
            else:
                insights[cat].append(f"{name} has become less predictable from one month to the next")

    # Limit to top 20 insights total, preserving category order
    total = sum(len(v) for v in insights.values())
    if total > 20:
        count = 0
        for key in list(insights.keys()):
            space = 20 - count
            if space <= 0:
                insights[key] = []
            else:
                insights[key] = insights[key][:space]
            count += len(insights[key])

    # Remove empty categories
    return {k: v for k, v in insights.items() if v}


def generate_trade_insights(
    tables: Dict[str, pd.DataFrame],
    trade_data: pd.DataFrame,
) -> Dict[str, List[str]]:
    """Generate narrative insights from merchandise trade tables and BOP trade flows.

    Returns a dict keyed by category ("Commodities", "Partners", "Trade Balance")
    with lists of insight strings, each category sorted by |z-score| descending,
    total capped at 20.
    """

    # Collect (abs_z, category, insight_text) tuples for sorting
    scored: List[Tuple[float, str, str]] = []

    # --- Merchandise table analysis ---
    table_meta = {
        "exp_commodity": ("Exports of", "Commodities"),
        "imp_commodity": ("Imports of", "Commodities"),
        "exp_country":   ("Exports to", "Partners"),
        "imp_country":   ("Imports from", "Partners"),
    }

    def _verb(pct: float) -> str:
        """Pick verb by magnitude of percentage change."""
        abs_pct = abs(pct)
        if pct > 0:
            if abs_pct > 50:
                return "surged"
            elif abs_pct > 20:
                return "grew strongly"
            else:
                return "grew notably"
        else:
            if abs_pct > 50:
                return "plunged"
            elif abs_pct > 20:
                return "fell sharply"
            else:
                return "declined notably"

    for key, (prefix, category) in table_meta.items():
        df = tables.get(key)
        if df is None or df.empty:
            continue

        # Take top 15 by trailing_4q for economic significance
        top = df.nlargest(15, "trailing_4q").copy()
        if len(top) < 3:
            continue

        # Track which items have been flagged (for dedup between YoY and QoQ)
        flagged: Dict[str, Tuple[float, str, str]] = {}  # name -> (abs_z, category, insight)

        # YoY outliers
        yoy = top["yoy_pct"].dropna()
        if len(yoy) >= 3:
            mean_yoy = yoy.mean()
            std_yoy = yoy.std()
            if std_yoy > 0:
                for _, row in top.iterrows():
                    if pd.isna(row["yoy_pct"]):
                        continue
                    z = (row["yoy_pct"] - mean_yoy) / std_yoy
                    if abs(z) > 1.5 and abs(row["yoy_pct"]) > 15:
                        verb = _verb(row["yoy_pct"])
                        insight = (
                            f"{prefix} {row['Name']} {verb} year-on-year "
                            f"({row['yoy_pct']:+.1f}%) to ${row['trailing_4q']:.1f}b "
                            f"over the trailing 4 quarters"
                        )
                        flagged[row["Name"]] = (abs(z), category, insight)

        # QoQ outliers
        qoq = top["qoq_pct"].dropna()
        if len(qoq) >= 3:
            mean_qoq = qoq.mean()
            std_qoq = qoq.std()
            if std_qoq > 0:
                q_label = top["_latest_q_label"].iloc[0] if "_latest_q_label" in top.columns else "latest quarter"
                for _, row in top.iterrows():
                    if pd.isna(row["qoq_pct"]):
                        continue
                    z = (row["qoq_pct"] - mean_qoq) / std_qoq
                    if abs(z) > 1.5 and abs(row["qoq_pct"]) > 15:
                        verb = _verb(row["qoq_pct"])
                        insight = (
                            f"{prefix} {row['Name']} {verb} quarter-on-quarter "
                            f"({row['qoq_pct']:+.1f}%) to ${row['latest_qtr']:.1f}b "
                            f"in {q_label}"
                        )
                        # Dedup: keep higher z-score entry
                        existing = flagged.get(row["Name"])
                        if existing is None or abs(z) > existing[0]:
                            flagged[row["Name"]] = (abs(z), category, insight)

        scored.extend(flagged.values())

    # --- Trade flow time series analysis (Goods Balance, Services Balance) ---
    for col_name in ["Goods Balance", "Services Balance"]:
        if col_name not in trade_data.columns:
            continue
        series = trade_data[col_name].dropna()
        if len(series) < 5:
            continue

        # Latest quarter label
        latest_q = trade_data["TIME_PERIOD"].iloc[-1] if "TIME_PERIOD" in trade_data.columns else "latest quarter"

        # Trend detection
        w = find_trend_window(series, min_window=4, max_window=40)
        if len(series) >= w:
            has_trend, desc, is_flat = detect_rolling_trend(series, window=w)
            if has_trend and desc and not is_flat:
                years, rem = divmod(w, 4)
                if rem == 0 and years >= 1:
                    window_desc = f"the past year" if years == 1 else f"the past {years} years"
                else:
                    window_desc = f"the past {w} quarters"
                if "decelerat" in desc:
                    insight = f"{col_name} has been decelerating — the recent trend is weaker than {window_desc} trend"
                elif "accelerat" in desc:
                    insight = f"{col_name} has been accelerating — recent momentum exceeds {window_desc} trend"
                else:
                    direction = "up" if "up" in desc else "down"
                    insight = f"{col_name} has been trending {direction} over {window_desc}"
                scored.append((2.0, "Trade Balance", insight))

        # Large latest-quarter swing
        is_large, z, change = detect_large_change(series, threshold=1.8)
        if is_large:
            direction = "widened" if change > 0 else "narrowed"
            insight = (
                f"{col_name} {direction} sharply by ${abs(change):.1f}bn "
                f"in {latest_q}"
            )
            scored.append((abs(z), "Trade Balance", insight))

    # Sort by |z-score| descending, cap at 20, group by category
    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:20]

    category_order = ["Commodities", "Partners", "Trade Balance"]
    result: Dict[str, List[str]] = {}
    for cat in category_order:
        items = [text for _, c, text in scored if c == cat]
        if items:
            result[cat] = items
    return result


def generate_services_trade_insights(
    tables: Dict[str, pd.DataFrame],
) -> Dict[str, List[str]]:
    """Generate narrative insights from services trade tables.

    Analyzes the 4 services summary tables for YoY outliers using z-scores.
    Returns a dict keyed by category ("Service Types", "Partners") with
    lists of insight strings, capped at 10.
    """

    scored: List[Tuple[float, str, str]] = []

    table_meta = {
        "svc_exp_type":    ("Services exports of", "Service Types"),
        "svc_imp_type":    ("Services imports of", "Service Types"),
        "svc_exp_country": ("Services exports to", "Partners"),
        "svc_imp_country": ("Services imports from", "Partners"),
    }

    def _verb(pct: float) -> str:
        abs_pct = abs(pct)
        if pct > 0:
            if abs_pct > 50:
                return "surged"
            elif abs_pct > 20:
                return "grew strongly"
            else:
                return "grew notably"
        else:
            if abs_pct > 50:
                return "plunged"
            elif abs_pct > 20:
                return "fell sharply"
            else:
                return "declined notably"

    for key, (prefix, category) in table_meta.items():
        df = tables.get(key)
        if df is None or df.empty:
            continue

        top = df.nlargest(10, "latest_yr").copy()
        if len(top) < 3:
            continue

        yoy = top["yoy_pct"].dropna()
        if len(yoy) < 3:
            continue
        mean_yoy = yoy.mean()
        std_yoy = yoy.std()
        if std_yoy <= 0:
            continue

        yr_label = top["_latest_yr_label"].iloc[0] if "_latest_yr_label" in top.columns else "latest year"

        for _, row in top.iterrows():
            if pd.isna(row["yoy_pct"]):
                continue
            z = (row["yoy_pct"] - mean_yoy) / std_yoy
            if abs(z) > 1.5 and abs(row["yoy_pct"]) > 15:
                verb = _verb(row["yoy_pct"])
                insight = (
                    f"{prefix} {row['Name']} {verb} year-on-year "
                    f"({row['yoy_pct']:+.1f}%) to ${row['latest_yr']:.1f}b "
                    f"in {yr_label}"
                )
                scored.append((abs(z), category, insight))

    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:10]

    category_order = ["Service Types", "Partners"]
    result: Dict[str, List[str]] = {}
    for cat in category_order:
        items = [text for _, c, text in scored if c == cat]
        if items:
            result[cat] = items
    return result


def generate_trade_tables_html(tables: Dict[str, pd.DataFrame]) -> str:
    """Build HTML for the 4 merchandise trade tables in a 2x2 grid."""

    titles = {
        "exp_commodity": "Top Goods Exports by Commodity",
        "imp_commodity": "Top Goods Imports by Commodity",
        "exp_country": "Top Goods Export Partners",
        "imp_country": "Top Goods Import Partners",
    }

    def _fmt_val(v, fmt="bn"):
        if pd.isna(v):
            return '<td class="tt-num">—</td>'
        if fmt == "bn":
            return f'<td class="tt-num">${v:,.1f}</td>'
        elif fmt == "pct":
            color = "#27ae60" if v >= 0 else "#e74c3c"
            return f'<td class="tt-num" style="color:{color}">{v:+.1f}%</td>'
        elif fmt == "share":
            return f'<td class="tt-num">{v:.1f}%</td>'
        return f'<td class="tt-num">{v}</td>'

    def _build_table(key):
        df = tables.get(key)
        if df is None or df.empty:
            return f'<div class="trade-table-wrap"><h3>{titles[key]}</h3><p>Data unavailable</p></div>'

        q_label = df["_latest_q_label"].iloc[0] if "_latest_q_label" in df.columns else "Latest"

        rows_html = ""
        for _, row in df.iterrows():
            rows_html += "<tr>"
            rows_html += f'<td class="tt-name" title="{row["Name"]}">{row["Name"]}</td>'
            rows_html += _fmt_val(row["latest_qtr"], "bn")
            rows_html += _fmt_val(row["trailing_4q"], "bn")
            rows_html += _fmt_val(row["qoq_pct"], "pct")
            rows_html += _fmt_val(row["yoy_pct"], "pct")
            rows_html += _fmt_val(row["share_pct"], "share")
            rows_html += "</tr>\n"

        return f"""<div class="trade-table-wrap">
            <h3>{titles[key]}</h3>
            <div class="trade-table-scroll">
                <table class="trade-table">
                    <thead>
                        <tr>
                            <th class="tt-name-hdr">Name</th>
                            <th class="tt-num-hdr">{q_label}<br>($bn)</th>
                            <th class="tt-num-hdr">Trail 4Q<br>($bn)</th>
                            <th class="tt-num-hdr">QoQ<br>(%)</th>
                            <th class="tt-num-hdr">YoY<br>(%)</th>
                            <th class="tt-num-hdr">Share<br>(%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            <p class="trade-table-source">Source: ABS Merchandise Trade (MERCH_EXP / MERCH_IMP)</p>
        </div>"""

    # 2x2 grid: exports left, imports right
    html = '<div class="trade-tables-grid">\n'
    html += _build_table("exp_commodity")
    html += _build_table("imp_commodity")
    html += _build_table("exp_country")
    html += _build_table("imp_country")
    html += "</div>\n"
    return html


def generate_services_tables_html(tables: Dict[str, pd.DataFrame]) -> str:
    """Build HTML for the 4 services trade tables in a 2x2 grid."""

    titles = {
        "svc_exp_type": "Top Services Exports by Type",
        "svc_imp_type": "Top Services Imports by Type",
        "svc_exp_country": "Top Services Export Partners",
        "svc_imp_country": "Top Services Import Partners",
    }

    def _fmt_val(v, fmt="bn"):
        if pd.isna(v):
            return '<td class="tt-num">—</td>'
        if fmt == "bn":
            return f'<td class="tt-num">${v:,.1f}</td>'
        elif fmt == "pct":
            color = "#27ae60" if v >= 0 else "#e74c3c"
            return f'<td class="tt-num" style="color:{color}">{v:+.1f}%</td>'
        elif fmt == "share":
            return f'<td class="tt-num">{v:.1f}%</td>'
        return f'<td class="tt-num">{v}</td>'

    def _build_table(key):
        df = tables.get(key)
        if df is None or df.empty:
            return f'<div class="trade-table-wrap"><h3>{titles[key]}</h3><p>Data unavailable</p></div>'

        yr_label = df["_latest_yr_label"].iloc[0] if "_latest_yr_label" in df.columns else "Latest"

        rows_html = ""
        for _, row in df.iterrows():
            rows_html += "<tr>"
            rows_html += f'<td class="tt-name" title="{row["Name"]}">{row["Name"]}</td>'
            rows_html += _fmt_val(row["latest_yr"], "bn")
            rows_html += _fmt_val(row["prior_yr"], "bn")
            rows_html += _fmt_val(row["yoy_pct"], "pct")
            rows_html += _fmt_val(row["share_pct"], "share")
            rows_html += "</tr>\n"

        return f"""<div class="trade-table-wrap">
            <h3>{titles[key]}</h3>
            <div class="trade-table-scroll">
                <table class="trade-table">
                    <thead>
                        <tr>
                            <th class="tt-name-hdr">Name</th>
                            <th class="tt-num-hdr">{yr_label}<br>($bn)</th>
                            <th class="tt-num-hdr">Prior Yr<br>($bn)</th>
                            <th class="tt-num-hdr">YoY<br>(%)</th>
                            <th class="tt-num-hdr">Share<br>(%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            <p class="trade-table-source">Source: ABS International Trade in Services (TRADE_SERV_CNTRY)</p>
        </div>"""

    html = '<div class="trade-tables-grid">\n'
    html += _build_table("svc_exp_type")
    html += _build_table("svc_imp_type")
    html += _build_table("svc_exp_country")
    html += _build_table("svc_imp_country")
    html += "</div>\n"
    return html


def create_contributions_chart(
    contributions: pd.DataFrame,
    gdp_growth: pd.DataFrame,
    title: str = "Contributions to GDP Growth (Quarterly)"
) -> go.Figure:
    """Create combined bar + line chart for GDP contributions."""

    # Merge data
    df = contributions.merge(gdp_growth, on="TIME_PERIOD", how="outer")
    df = df.sort_values("TIME_PERIOD").reset_index(drop=True)

    # Create figure
    fig = go.Figure()

    # Add bars for each component
    components = ["Consumption", "GFCF", "Inventories", "Exports", "Imports"]
    for comp in components:
        if comp in df.columns:
            fig.add_trace(go.Bar(
                name=comp,
                x=df["TIME_PERIOD"],
                y=df[comp],
                marker_color=GDP_COLORS[comp],
                hovertemplate=f"{comp}: %{{y:.1f}} ppts<extra></extra>",
            ))

    # Add GDP growth line
    fig.add_trace(go.Scatter(
        name="GDP Growth",
        x=df["TIME_PERIOD"],
        y=df["GDP_growth"],
        mode="lines+markers",
        line=dict(color="black", width=2),
        marker=dict(size=6, color="black"),
        hovertemplate="GDP: %{y:.1f}%<extra></extra>",
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18),
        ),
        barmode="relative",  # Stacked bars (positive/negative)
        xaxis=dict(
            title="Quarter",
            tickangle=-45,
            dtick=4,  # Show every 4th tick (yearly)
        ),
        yaxis=dict(
            title="Percentage Points",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        template="plotly_white",
        hovermode="x unified",
        height=500,
        margin=dict(l=60, r=40, t=100, b=80),
    )

    # Add annotation for data source
    fig.add_annotation(
        text="Source: ABS National Accounts (Cat. 5206.0)",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.18,
        showarrow=False,
        font=dict(size=10, color="gray"),
    )

    return fig


def create_current_account_chart(
    ca_data: pd.DataFrame,
    title: str = "Current Account Balance (Quarterly)"
) -> go.Figure:
    """Create stacked bar chart for current account components."""

    df = ca_data.copy()

    # Create figure
    fig = go.Figure()

    # Add bars for each component
    components = ["Goods", "Services", "Primary Income", "Secondary Income"]
    for comp in components:
        if comp in df.columns:
            fig.add_trace(go.Bar(
                name=comp,
                x=df["TIME_PERIOD"],
                y=df[comp],
                marker_color=CA_COLORS[comp],
                hovertemplate=f"{comp}: $%{{y:.1f}}bn<extra></extra>",
            ))

    # Add current account balance line
    if "Current Account" in df.columns:
        fig.add_trace(go.Scatter(
            name="Current Account",
            x=df["TIME_PERIOD"],
            y=df["Current Account"],
            mode="lines+markers",
            line=dict(color="black", width=2),
            marker=dict(size=6, color="black"),
            hovertemplate="CA: $%{y:.1f}bn<extra></extra>",
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18),
        ),
        barmode="relative",  # Stacked bars (positive/negative)
        xaxis=dict(
            title="Quarter",
            tickangle=-45,
            dtick=4,  # Show every 4th tick (yearly)
        ),
        yaxis=dict(
            title="A$ Billion",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        template="plotly_white",
        hovermode="x unified",
        height=500,
        margin=dict(l=60, r=40, t=100, b=80),
    )

    # Add annotation for data source
    fig.add_annotation(
        text="Source: ABS Balance of Payments (Cat. 5302.0)",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.18,
        showarrow=False,
        font=dict(size=10, color="gray"),
    )

    return fig


def create_trade_chart(
    trade_data: pd.DataFrame,
) -> List[go.Figure]:
    """Create individual charts for Goods and Services trade flows.

    Each chart shows credit/debit lines and a balance bar.
    Returns a list of two figures: [goods_fig, services_fig].
    """

    df = trade_data.copy()
    _common = dict(
        barmode="relative", showlegend=False, template="plotly_white",
        hovermode="x unified", height=420, margin=dict(l=60, r=30, t=50, b=80),
    )

    # --- Goods chart ---
    goods_fig = go.Figure()
    if "Goods Balance" in df.columns:
        goods_fig.add_trace(go.Bar(
            name="Goods Balance",
            x=df["TIME_PERIOD"],
            y=df["Goods Balance"],
            marker_color=TRADE_COLORS["Goods Balance"],
            marker_opacity=0.45,
            hovertemplate="Goods Balance: $%{y:.1f}bn<extra></extra>",
        ))

    for comp in ["Goods Credits", "Goods Debits"]:
        if comp in df.columns:
            goods_fig.add_trace(go.Scatter(
                name=comp,
                x=df["TIME_PERIOD"],
                y=df[comp],
                mode="lines",
                line=dict(color=TRADE_COLORS[comp], width=2),
                hovertemplate=f"{comp}: $%{{y:.1f}}bn<extra></extra>",
            ))

    goods_fig.update_layout(**_common, title="Goods")
    goods_fig.update_xaxes(tickangle=-45, dtick=4)
    goods_fig.update_yaxes(title_text="A$ Billion", zeroline=True, zerolinewidth=1, zerolinecolor="gray")

    # --- Services chart ---
    services_fig = go.Figure()
    if "Services Balance" in df.columns:
        services_fig.add_trace(go.Bar(
            name="Services Balance",
            x=df["TIME_PERIOD"],
            y=df["Services Balance"],
            marker_color=TRADE_COLORS["Services Balance"],
            marker_opacity=0.45,
            hovertemplate="Services Balance: $%{y:.1f}bn<extra></extra>",
        ))

    for comp in ["Services Credits", "Services Debits"]:
        if comp in df.columns:
            services_fig.add_trace(go.Scatter(
                name=comp,
                x=df["TIME_PERIOD"],
                y=df[comp],
                mode="lines",
                line=dict(color=TRADE_COLORS[comp], width=2),
                hovertemplate=f"{comp}: $%{{y:.1f}}bn<extra></extra>",
            ))

    services_fig.update_layout(**_common, title="Services")
    services_fig.update_xaxes(tickangle=-45, dtick=4)
    services_fig.update_yaxes(title_text="A$ Billion", zeroline=True, zerolinewidth=1, zerolinecolor="gray")

    return [goods_fig, services_fig]


def create_dashboard(start_period: str = "2015-Q1") -> Tuple[List[go.Figure], Dict[str, pd.DataFrame]]:
    """Create a dashboard with GDP, Current Account, Inflation, and Labour Market charts.

    Returns:
        Tuple of (list of individual figures, dict of dataframes for insight generation)
    """

    # Fetch all data
    print("Fetching GDP contribution data...")
    contributions = get_gdp_contributions(start_period)
    gdp_growth = get_gdp_growth(start_period)

    print("Fetching current account data...")
    ca_data = get_current_account(start_period)

    print("Fetching trade flow data...")
    trade_data = get_trade_data(start_period)

    print("Fetching inflation data from RBA...")
    start_year = int(start_period.split("-")[0])
    inflation_data = get_inflation_rba(start_year)

    print("Fetching labour force data...")
    # Convert quarterly start to monthly for labour force
    lf_start = start_period.replace("-Q1", "-01").replace("-Q2", "-04").replace("-Q3", "-07").replace("-Q4", "-10")
    lf_data = get_labour_force(lf_start)

    # Merge GDP data
    gdp_df = contributions.merge(gdp_growth, on="TIME_PERIOD", how="outer")
    gdp_df = gdp_df.sort_values("TIME_PERIOD").reset_index(drop=True)

    # Shared layout settings
    _common = dict(template="plotly_white", hovermode="x unified", showlegend=False)

    # ===== 1. GDP Contributions =====
    gdp_fig = go.Figure()
    gdp_components = ["Consumption", "GFCF", "Inventories", "Exports", "Imports"]
    for comp in gdp_components:
        if comp in gdp_df.columns:
            gdp_fig.add_trace(go.Bar(
                name=comp,
                x=gdp_df["TIME_PERIOD"],
                y=gdp_df[comp],
                marker_color=GDP_COLORS[comp],
                hovertemplate=f"{comp}: %{{y:.1f}} ppts<extra></extra>",
            ))

    gdp_fig.add_trace(go.Scatter(
        name="GDP Growth",
        x=gdp_df["TIME_PERIOD"],
        y=gdp_df["GDP_growth"],
        mode="lines+markers",
        line=dict(color="black", width=2),
        marker=dict(size=4, color="black"),
        hovertemplate="GDP: %{y:.1f}%<extra></extra>",
    ))

    gdp_fig.update_layout(
        **_common, barmode="relative", title="Contributions to GDP Growth",
        height=420, margin=dict(l=60, r=30, t=50, b=80),
    )
    gdp_fig.update_xaxes(tickangle=-45, dtick=4)
    gdp_fig.update_yaxes(title_text="ppts", zeroline=True, zerolinewidth=1)

    # ===== 2. Current Account =====
    ca_fig = go.Figure()
    ca_components = ["Goods", "Services", "Primary Income", "Secondary Income"]
    for comp in ca_components:
        if comp in ca_data.columns:
            ca_fig.add_trace(go.Bar(
                name=comp,
                x=ca_data["TIME_PERIOD"],
                y=ca_data[comp],
                marker_color=CA_COLORS[comp],
                hovertemplate=f"{comp}: $%{{y:.1f}}bn<extra></extra>",
            ))

    if "Current Account" in ca_data.columns:
        ca_fig.add_trace(go.Scatter(
            name="Current Account",
            x=ca_data["TIME_PERIOD"],
            y=ca_data["Current Account"],
            mode="lines+markers",
            line=dict(color="black", width=2),
            marker=dict(size=4, color="black"),
            hovertemplate="CA: $%{y:.1f}bn<extra></extra>",
        ))

    ca_fig.update_layout(
        **_common, barmode="relative", title="Current Account Balance",
        height=420, margin=dict(l=60, r=30, t=50, b=80),
    )
    ca_fig.update_xaxes(tickangle=-45, dtick=4)
    ca_fig.update_yaxes(title_text="A$bn", zeroline=True, zerolinewidth=1)

    # ===== 3. Inflation =====
    inf_fig = go.Figure()
    if "Headline" in inflation_data.columns:
        inf_fig.add_trace(go.Scatter(
            name="Headline CPI",
            x=inflation_data["TIME_PERIOD"],
            y=inflation_data["Headline"],
            mode="lines",
            line=dict(color="#AAAAAA", width=1.5),
            hovertemplate="Headline: %{y:.1f}%<extra></extra>",
        ))

    if "Trimmed Mean" in inflation_data.columns:
        inf_fig.add_trace(go.Scatter(
            name="Trimmed Mean",
            x=inflation_data["TIME_PERIOD"],
            y=inflation_data["Trimmed Mean"],
            mode="lines",
            line=dict(color="#E45756", width=2.5),
            hovertemplate="Trimmed Mean: %{y:.1f}%<extra></extra>",
        ))

    inf_fig.add_hrect(y0=2, y1=3, line_width=0, fillcolor="rgba(0,128,0,0.1)")
    inf_fig.add_hline(y=2, line_dash="dash", line_color="green", line_width=1)
    inf_fig.add_hline(y=3, line_dash="dash", line_color="green", line_width=1)

    inf_fig.update_layout(
        **_common, title="Inflation",
        height=380, margin=dict(l=50, r=30, t=50, b=80),
    )
    inf_fig.update_xaxes(tickangle=-45, dtick=4)
    inf_fig.update_yaxes(title_text="%", zeroline=True, zerolinewidth=1)

    # ===== 4. Unemployment Rate =====
    unemp_fig = go.Figure()
    if "Unemployment Rate" in lf_data.columns:
        unemp_fig.add_trace(go.Scatter(
            name="Unemployment Rate",
            x=lf_data["TIME_PERIOD"],
            y=lf_data["Unemployment Rate"],
            mode="lines",
            line=dict(color="#E45756", width=2),
            hovertemplate="Unemployment: %{y:.1f}%<extra></extra>",
        ))

    unemp_fig.update_layout(
        **_common, title="Unemployment Rate",
        height=380, margin=dict(l=50, r=30, t=50, b=80),
    )
    unemp_fig.update_xaxes(tickangle=-45, dtick=12)
    unemp_fig.update_yaxes(title_text="%", zeroline=True, zerolinewidth=1)

    # ===== 5. Participation & Employment =====
    part_fig = go.Figure()
    if "Participation Rate" in lf_data.columns:
        part_fig.add_trace(go.Scatter(
            name="Participation Rate",
            x=lf_data["TIME_PERIOD"],
            y=lf_data["Participation Rate"],
            mode="lines",
            line=dict(color="#4C78A8", width=2),
            hovertemplate="Participation: %{y:.1f}%<extra></extra>",
        ))

    if "Employment/Pop Ratio" in lf_data.columns:
        part_fig.add_trace(go.Scatter(
            name="Employment/Pop Ratio",
            x=lf_data["TIME_PERIOD"],
            y=lf_data["Employment/Pop Ratio"],
            mode="lines",
            line=dict(color="#54A24B", width=2),
            hovertemplate="Emp/Pop: %{y:.1f}%<extra></extra>",
        ))

    part_fig.update_layout(
        **_common, title="Participation & Employment",
        height=380, margin=dict(l=50, r=30, t=50, b=80),
    )
    part_fig.update_xaxes(tickangle=-45, dtick=12)
    part_fig.update_yaxes(title_text="%")

    charts = [gdp_fig, ca_fig, inf_fig, unemp_fig, part_fig]

    # Return individual figures and data for insight generation
    data = {
        "gdp": gdp_df,
        "ca": ca_data,
        "inflation": inflation_data,
        "labour": lf_data,
        "trade": trade_data,
    }

    return charts, data


def create_html_with_insights(
    charts: List[go.Figure],
    insights: dict,
    output_path: str = "dashboard.html",
    trade_figs: List[go.Figure] = None,
    trade_tables_html: str = "",
    trade_insights: Dict[str, List[str]] = None,
    services_tables_html: str = "",
    services_trade_insights: Dict[str, List[str]] = None,
):
    """Create HTML file with dashboard and insights summary box.

    charts is a list of individual Plotly figures for the Big Picture tab,
    rendered in a responsive CSS grid (2-col desktop, 1-col mobile).
    When trade_figs is provided, renders a tab bar (Big Picture / Trade) with
    pure CSS/JS toggle.
    """

    # Generate the plotly HTML for each chart (div only, not full page)
    chart_htmls = []
    for i, fig in enumerate(charts):
        html = fig.to_html(
            full_html=False,
            include_plotlyjs='cdn' if i == 0 else False,
            config={'responsive': True},
        )
        chart_htmls.append(html)

    # Build the grid of charts
    # Row 1: 2 wide charts (GDP, Current Account) — class "row-2"
    # Row 2: 3 charts (Inflation, Unemployment, Participation) — class "row-3"
    row1_panels = "\n".join(
        f'<div class="chart-panel"><div class="chart-container">{h}</div></div>'
        for h in chart_htmls[:2]
    )
    row2_panels = "\n".join(
        f'<div class="chart-panel"><div class="chart-container">{h}</div></div>'
        for h in chart_htmls[2:]
    )
    chart_grid_html = f"""
    <div class="charts-row charts-row-2">{row1_panels}</div>
    <div class="charts-row charts-row-3">{row2_panels}</div>
    <p class="source-note">Source: ABS National Accounts, Balance of Payments, Labour Force; RBA Statistical Table G1</p>
    """

    # Create grouped bullet points HTML with category subheadings
    if insights:
        sections = []
        for category, items in insights.items():
            if items:
                bullets = "\n".join(f"<li>{item}</li>" for item in items)
                sections.append(f"<h3>{category}</h3>\n<ul>\n{bullets}\n</ul>")
        bullets_html = "\n".join(sections) if sections else "<p>No unusual developments detected in the recent data.</p>"
    else:
        bullets_html = "<p>No unusual developments detected in the recent data.</p>"

    # Build tab bar and trade content if trade_figs is provided
    if trade_figs is not None:
        trade_panel_htmls = []
        for tfig in trade_figs:
            trade_panel_htmls.append(
                tfig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
            )
        trade_grid_panels = "\n".join(
            f'<div class="chart-panel"><div class="chart-container">{h}</div></div>'
            for h in trade_panel_htmls
        )
        trade_chart_grid = f"""
        <div class="charts-row charts-row-2">{trade_grid_panels}</div>
        <p class="source-note">Source: ABS Balance of Payments (Cat. 5302.0)</p>
        """

        tab_bar_html = """
        <div class="tab-bar">
            <button class="tab-btn active" onclick="switchTab('overview')">Big Picture</button>
            <button class="tab-btn" onclick="switchTab('trade')">Trade</button>
        </div>"""

        tab_css = """
        .tab-bar {
            display: flex;
            gap: 0;
            margin-bottom: 0;
        }
        .tab-btn {
            padding: 12px 28px;
            border: 1px solid #dee2e6;
            border-bottom: none;
            background: #e9ecef;
            color: #495057;
            font-size: 1em;
            font-weight: 500;
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            font-family: inherit;
            transition: background 0.15s, color 0.15s;
        }
        .tab-btn:hover {
            background: #f8f9fa;
        }
        .tab-btn.active {
            background: white;
            color: #2c3e50;
            font-weight: 600;
            border-bottom: 2px solid white;
            margin-bottom: -1px;
            position: relative;
            z-index: 1;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .trade-tables-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-top: 28px;
        }
        .trade-table-wrap {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            padding: 16px;
            border: 1px solid #dee2e6;
        }
        .trade-table-wrap h3 {
            margin: 0 0 12px 0;
            color: #2c3e50;
            font-size: 1.05em;
            font-weight: 600;
        }
        .trade-table-scroll {
            max-height: 420px;
            overflow-y: auto;
        }
        .trade-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88em;
        }
        .trade-table thead {
            position: sticky;
            top: 0;
            z-index: 1;
        }
        .trade-table th {
            background: #f1f3f5;
            color: #495057;
            font-weight: 600;
            padding: 8px 10px;
            text-align: right;
            border-bottom: 2px solid #dee2e6;
            white-space: nowrap;
        }
        .trade-table th.tt-name-hdr {
            text-align: left;
        }
        .trade-table td {
            padding: 6px 10px;
            border-bottom: 1px solid #eee;
        }
        .trade-table tr:hover {
            background: #f8f9fa;
        }
        .tt-num {
            text-align: right;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }
        .tt-name {
            text-align: left;
            max-width: 180px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .trade-table-source {
            margin: 8px 0 0 0;
            font-size: 0.78em;
            color: #999;
        }
        .trade-section-heading {
            margin: 32px 0 4px 0;
            color: #2c3e50;
            font-size: 1.3em;
            font-weight: 600;
        }"""

        tab_js = """
    <script>
    function switchTab(tabName) {
        // Hide all tab content
        document.querySelectorAll('.tab-content').forEach(function(el) {
            el.classList.remove('active');
        });
        // Deactivate all tab buttons
        document.querySelectorAll('.tab-btn').forEach(function(el) {
            el.classList.remove('active');
        });
        // Show selected tab content
        document.getElementById('tab-' + tabName).classList.add('active');
        // Activate selected tab button
        event.target.classList.add('active');
        // Dispatch resize so Plotly redraws on hidden->visible
        window.dispatchEvent(new Event('resize'));
    }
    </script>"""

        # Build 3 separate trade insight boxes
        def _build_insight_box(insight_dict):
            if not insight_dict:
                return ""
            sections = []
            for cat, items in insight_dict.items():
                if items:
                    bullets = "\n".join(f"<li>{item}</li>" for item in items)
                    sections.append(f"<h3>{cat}</h3>\n<ul>\n{bullets}\n</ul>")
            if not sections:
                return ""
            return f"""
            <div class="insights-box">
                {"".join(sections)}
            </div>"""

        # 1) Chart insights: only "Trade Balance" from trade_insights
        chart_insight_dict = {}
        if trade_insights and "Trade Balance" in trade_insights:
            chart_insight_dict["Trade Balance"] = trade_insights["Trade Balance"]
        chart_insights_html = _build_insight_box(chart_insight_dict)

        # 2) Goods table insights: "Commodities" + "Partners" from trade_insights
        goods_insight_dict = {}
        if trade_insights:
            for cat in ["Commodities", "Partners"]:
                if cat in trade_insights:
                    goods_insight_dict[cat] = trade_insights[cat]
        goods_insights_html = _build_insight_box(goods_insight_dict)

        # 3) Services insights
        services_insights_html = _build_insight_box(services_trade_insights)

        body_content = f"""
        {tab_bar_html}
        <div id="tab-overview" class="tab-content active">
            {chart_grid_html}
            <div class="insights-box">
                {bullets_html}
            </div>
        </div>
        <div id="tab-trade" class="tab-content">
            {trade_chart_grid}
            {chart_insights_html}
            <h2 class="trade-section-heading">Goods Trade</h2>
            {trade_tables_html}
            {goods_insights_html}
            <h2 class="trade-section-heading">Services Trade</h2>
            {services_tables_html}
            {services_insights_html}
        </div>
        {tab_js}"""
    else:
        tab_css = ""
        body_content = f"""
        {chart_grid_html}
        <div class="insights-box">
            {bullets_html}
        </div>"""

    # Full HTML template
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Australian Macroeconomic Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #fafafa;
        }}
        .container {{
            max-width: 1450px;
            margin: 0 auto;
        }}
        .insights-box {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 1px solid #dee2e6;
            border-left: 4px solid #4C78A8;
            border-radius: 8px;
            padding: 20px 25px;
            margin-top: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .insights-box h2 {{
            margin: 0 0 15px 0;
            color: #2c3e50;
            font-size: 1.3em;
            font-weight: 600;
        }}
        .insights-box ul {{
            margin: 0;
            padding-left: 20px;
            color: #495057;
        }}
        .insights-box li {{
            margin-bottom: 8px;
            line-height: 1.5;
        }}
        .insights-box li:last-child {{
            margin-bottom: 0;
        }}
        .insights-box h3 {{
            margin: 15px 0 8px 0;
            color: #34495e;
            font-size: 1.05em;
            font-weight: 600;
        }}
        .insights-box h3:first-of-type {{
            margin-top: 0;
        }}
        .charts-row {{
            display: grid;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .charts-row-2 {{
            grid-template-columns: 1fr 1fr;
        }}
        .charts-row-3 {{
            grid-template-columns: 1fr 1fr 1fr;
        }}
        .chart-panel {{
            min-width: 0;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            padding: 10px;
            width: 100%;
            overflow-x: auto;
            box-sizing: border-box;
        }}
        .source-note {{
            text-align: center;
            font-size: 0.78em;
            color: #999;
            margin: 0 0 10px 0;
        }}
        .trade-chart-wrap {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px 0;
        }}
        {tab_css}
        @media (max-width: 768px) {{
            .container {{
                max-width: 100%;
                padding: 10px;
            }}
            body {{
                padding: 10px;
            }}
            .charts-row-2,
            .charts-row-3 {{
                grid-template-columns: 1fr;
            }}
            .trade-tables-grid {{
                grid-template-columns: 1fr;
            }}
            .trade-chart-wrap {{
                max-width: 100%;
            }}
            .insights-box {{
                padding: 15px 18px;
                margin-top: 20px;
            }}
            .tab-btn {{
                padding: 10px 20px;
                font-size: 0.92em;
            }}
            .trade-table {{
                font-size: 0.82em;
            }}
            .tt-name {{
                max-width: 130px;
            }}
        }}
        @media (max-width: 480px) {{
            body {{
                padding: 5px;
            }}
            .container {{
                padding: 5px;
            }}
            .chart-container {{
                padding: 5px;
            }}
            .insights-box {{
                padding: 12px 14px;
                margin-top: 15px;
            }}
            .tab-btn {{
                padding: 8px 14px;
                font-size: 0.85em;
            }}
            .trade-table {{
                font-size: 0.78em;
            }}
            .tt-name {{
                max-width: 100px;
            }}
            .trade-table th {{
                padding: 6px 6px;
            }}
            .trade-table td {{
                padding: 4px 6px;
            }}
            .trade-section-heading {{
                font-size: 1.1em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {body_content}
    </div>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html_template)


def main():
    """Main entry point."""
    print("Fetching data from ABS...")

    start_period = "2015-Q1"

    # Create dashboard charts and get data
    charts, data = create_dashboard(start_period)

    # Generate insights
    print("Generating insights...")
    insights = generate_insights(
        gdp_df=data["gdp"],
        ca_data=data["ca"],
        inflation_data=data["inflation"],
        lf_data=data["labour"]
    )

    # Print insights to console
    if insights:
        print("\nInsights:")
        for category, items in insights.items():
            print(f"\n  {category}:")
            for insight in items:
                print(f"    • {insight}")
    else:
        print("\nNo unusual developments detected.")

    # Create trade charts
    trade_charts = create_trade_chart(data["trade"])

    # Fetch merchandise trade tables
    trade_tables_data = get_merch_trade_tables("2023-01")
    trade_tables_html = generate_trade_tables_html(trade_tables_data)

    # Fetch services trade tables
    services_tables_data = get_services_trade_tables()
    services_tables_html = generate_services_tables_html(services_tables_data)

    # Generate trade insights
    print("Generating trade insights...")
    trade_insights = generate_trade_insights(trade_tables_data, data["trade"])

    if trade_insights:
        print("\nTrade Insights:")
        for category, items in trade_insights.items():
            print(f"\n  {category}:")
            for insight in items:
                print(f"    • {insight}")

    # Generate services trade insights
    print("Generating services trade insights...")
    services_trade_insights = generate_services_trade_insights(services_tables_data)

    if services_trade_insights:
        print("\nServices Trade Insights:")
        for category, items in services_trade_insights.items():
            print(f"\n  {category}:")
            for insight in items:
                print(f"    • {insight}")

    # Save HTML with insights
    create_html_with_insights(
        charts, insights, "dashboard.html",
        trade_figs=trade_charts, trade_tables_html=trade_tables_html,
        trade_insights=trade_insights,
        services_tables_html=services_tables_html,
        services_trade_insights=services_trade_insights,
    )
    print("\nSaved: dashboard.html")

    # Open dashboard in browser (skip in CI)
    import os
    if not os.environ.get("CI"):
        import webbrowser
        webbrowser.open("dashboard.html")


if __name__ == "__main__":
    main()
