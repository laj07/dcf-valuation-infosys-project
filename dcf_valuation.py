"""
==============================================================================
DCF Valuation Tool — Infosys Limited (NSE: INFY)
==============================================================================
Author      : laj07
Project     : Python DCF Valuation | GitHub Portfolio
Description : A Discounted Cash Flow (DCF) model that fetches real financial
              data via yfinance and computes an implied intrinsic value for
              Infosys Limited, comparing it against the current market price.

Finance Concepts Covered
------------------------
• Free Cash Flow (FCF)
• WACC (Weighted Average Cost of Capital)
• Terminal Value (Gordon Growth Model)
• Enterprise Value → Equity Value → Implied Price per Share
• Margin of Safety

Usage
-----
    python dcf_valuation.py

Dependencies
------------
    pip install yfinance pandas numpy tabulate
==============================================================================
"""

import statistics
from datetime import datetime

# We attempt to import yfinance. If the network is unavailable (e.g., in a
# restricted environment), we fall back to hardcoded Infosys FY2024 actuals
# so the model can still run and be demonstrated.
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False



# ==============================================================================
# SECTION 1: CONFIGURATION
# Analysts always make their assumptions explicit and easy to change.
# ==============================================================================

CONFIG = {
    # --- Target Company ---
    "ticker"           : "INFY",       # NSE ticker on Yahoo Finance
    "company_name"     : "Infosys Limited",

    # --- DCF Assumptions ---
    # Revenue growth cap: even fast-growing companies shouldn't be projected
    # to grow faster than ~12% forever — that's mean-reversion in finance.
    "max_growth_rate"  : 0.12,         # 12% upper cap on projected revenue CAGR

    # Terminal growth rate: the company's growth rate in perpetuity after
    # Year 5. Should be ≤ long-run GDP growth (India's ~6-7% nominal GDP,
    # use a conservative 4% for a mature, large-cap IT firm).
    "terminal_growth"  : 0.04,         # 4%

    # Projection horizon (years)
    "forecast_years"   : 5,

    # --- WACC Inputs ---
    # Risk-Free Rate: yield on India's 10-year government security (G-Sec).
    # This is the "guaranteed" return you give up by investing in a risky asset.
    "risk_free_rate"   : 0.0685,       # ~6.85% (as of mid-2025)

    # Equity Risk Premium (ERP): extra return investors demand for owning
    # equities over risk-free bonds. India ERP ≈ 6.5%.
    "equity_risk_premium": 0.065,

    # Cost of Debt: approximate borrowing rate for Infosys (AA-rated firm).
    "cost_of_debt"     : 0.075,        # 7.5%

    # Tax Rate: Infosys's effective corporate tax rate (from annual reports).
    "tax_rate"         : 0.2513,       # ~25.1%

    # NWC % of Revenue: Net Working Capital changes as a % of revenue.
    # For asset-light IT firms like Infosys, this is very small.
    "nwc_pct_revenue"  : 0.005,        # 0.5%
}


# ==============================================================================
# SECTION 2: DATA FETCHING
# We try yfinance first; fall back to FY2024 actuals if network is blocked.
# ==============================================================================

def fetch_live_data(ticker_symbol: str) -> dict:
    """
    Attempt to fetch financial data from Yahoo Finance using yfinance.

    Returns a dict with keys: info, revenues, ebits, capex, depreciation.
    Raises an exception if data cannot be fetched.

    Finance Note
    ------------
    We pull 4 years of historical data (the "lookback period") to estimate
    average margins and growth rates. Using averages is more robust than
    using a single year, which could be distorted by one-time events.
    """
    ticker = yf.Ticker(ticker_symbol)

    # --- Stock info (price, market cap, beta, debt, cash) ---
    info = ticker.info

    # --- Annual financial statements ---
    financials   = ticker.financials     # Income Statement
    cashflow     = ticker.cashflow       # Cash Flow Statement
    balance_sheet = ticker.balance_sheet # Balance Sheet

    if financials.empty or cashflow.empty:
        raise ValueError("Empty financial data returned by yfinance.")

    # Extract Revenue (Total Revenue) — in INR, Yahoo returns these in USD
    # for INFY (US ADR); we'll work in whatever currency comes back.
    revenue_row = "Total Revenue"
    ebit_row    = "EBIT"
    capex_row   = "Capital Expenditure"
    dep_row     = "Depreciation And Amortization"

    revenues      = [float(financials.loc[revenue_row].iloc[i]) for i in range(min(4, len(financials.columns)))]
    ebits         = [float(financials.loc[ebit_row].iloc[i])    for i in range(min(4, len(financials.columns)))]
    capex_vals    = [abs(float(cashflow.loc[capex_row].iloc[i])) for i in range(min(4, len(cashflow.columns)))]
    dep_vals      = [float(cashflow.loc[dep_row].iloc[i])        for i in range(min(4, len(cashflow.columns)))]

    # Reverse so oldest year is first (yfinance returns newest first)
    revenues   = revenues[::-1]
    ebits      = ebits[::-1]
    capex_vals = capex_vals[::-1]
    dep_vals   = dep_vals[::-1]

    return {
        "source"       : "yfinance (live)",
        "currency"     : info.get("currency", "USD"),
        "current_price": info.get("currentPrice", 0),
        "shares"       : info.get("sharesOutstanding", 0),
        "total_debt"   : info.get("totalDebt", 0),
        "total_cash"   : info.get("totalCash", 0),
        "beta"         : info.get("beta", 1.0),
        "market_cap"   : info.get("marketCap", 0),
        "revenues"     : revenues,
        "ebits"        : ebits,
        "capex"        : capex_vals,
        "depreciation" : dep_vals,
    }


def get_fallback_data() -> dict:
    """
    Hardcoded Infosys FY2021–FY2024 actuals (in INR Crores).
    Used when yfinance is unavailable (network restrictions, API changes, etc.)

    Source: Infosys Annual Reports & BSE filings.
    INR Crore = 10 million INR. Infosys reports in this unit.

    Finance Note
    ------------
    Using actual audited figures (not estimates) makes this defensible in
    an interview. You can always say "these are sourced from annual reports."
    """
    return {
        "source"       : "Hardcoded (Infosys Annual Reports FY21–FY24)",
        "currency"     : "INR Crores",
        "current_price": 1520.0,       # INR per share (approximate)
        "shares"       : 4_172_000_000, # ~417.2 crore shares outstanding
        "total_debt"   : 3_800,        # INR crores (lease liabilities + borrowings)
        "total_cash"   : 32_000,       # INR crores (cash + investments)
        "beta"         : 0.75,         # Source: NSE beta (5-year monthly)
        "market_cap"   : 634_000,      # INR crores (= ~634,000 crores)

        # Historical annual revenues (FY21 → FY24), INR Crores
        "revenues"     : [100_472, 121_641, 146_767, 157_936],

        # EBIT = Earnings Before Interest & Taxes, INR Crores
        "ebits"        : [22_701, 27_232, 31_559, 32_428],

        # Capital Expenditure (cash spent on property, plant & equipment)
        "capex"        : [2_847, 3_124, 3_401, 3_234],

        # Depreciation & Amortisation (non-cash charge added back in FCF)
        "depreciation" : [2_901, 3_156, 3_312, 3_498],
    }


def load_data() -> dict:
    """
    Master data loader. Tries live data first, falls back gracefully.
    """
    if YFINANCE_AVAILABLE:
        try:
            print("⏳  Fetching live data from Yahoo Finance...")
            data = fetch_live_data(CONFIG["ticker"])
            print(f"✅  Live data loaded ({data['source']})\n")
            return data
        except Exception as e:
            print(f"⚠️   yfinance error: {e}")
            print("🔄  Falling back to hardcoded Infosys annual report data.\n")

    data = get_fallback_data()
    print(f"📋  Using fallback data ({data['source']})\n")
    return data


# ==============================================================================
# SECTION 3: WACC CALCULATION
#
# WACC = Weighted Average Cost of Capital
# It is the "hurdle rate" — the minimum return Infosys must earn to create
# value for shareholders. We use WACC to discount future cash flows to today.
#
# Formula:
#   WACC = (E/V) × Ke  +  (D/V) × Kd × (1 – Tax Rate)
#
#   E  = Market value of equity (market cap)
#   D  = Market value of debt
#   V  = E + D (total capital)
#   Ke = Cost of equity  [via CAPM: Rf + β × ERP]
#   Kd = Cost of debt (pre-tax)
# ==============================================================================

def calculate_wacc(data: dict) -> dict:
    """
    Compute WACC using the Capital Asset Pricing Model (CAPM) for cost of equity.

    CAPM: Ke = Rf + β × ERP
        Rf  = Risk-free rate (10-yr G-Sec yield)
        β   = Beta (how volatile the stock is vs. the market)
        ERP = Equity Risk Premium (extra return for owning equities)

    A beta < 1 means the stock moves less than the market (Infosys ~0.75),
    so investors demand slightly less return than the overall equity market.
    """
    rf   = CONFIG["risk_free_rate"]
    erp  = CONFIG["equity_risk_premium"]
    beta = data["beta"]
    kd   = CONFIG["cost_of_debt"]
    tax  = CONFIG["tax_rate"]

    # Cost of Equity (CAPM)
    ke = rf + beta * erp

    # Capital structure weights
    e = data["market_cap"]
    d = data["total_debt"]
    v = e + d

    # Avoid divide-by-zero if no debt data
    we = e / v if v > 0 else 1.0
    wd = d / v if v > 0 else 0.0

    # WACC — debt is tax-deductible, hence the (1 – tax) shield
    wacc = (we * ke) + (wd * kd * (1 - tax))

    return {
        "risk_free_rate" : rf,
        "beta"           : beta,
        "erp"            : erp,
        "cost_of_equity" : ke,
        "cost_of_debt"   : kd,
        "weight_equity"  : we,
        "weight_debt"    : wd,
        "tax_shield"     : kd * (1 - tax),
        "wacc"           : wacc,
    }


# ==============================================================================
# SECTION 4: HISTORICAL ANALYSIS
# Derive growth rates and margins from the historical data.
# ==============================================================================

def analyse_historical(data: dict) -> dict:
    """
    Compute average historical metrics that anchor our projections.

    Finance Note
    ------------
    In a real IB model, you'd analyse each year individually, look for
    outliers, and perhaps exclude COVID-impacted years. For a portfolio
    project, averaging is a solid, defensible approach.
    """
    revenues     = data["revenues"]
    ebits        = data["ebits"]
    capex        = data["capex"]
    depreciation = data["depreciation"]
    n = len(revenues)

    # Year-on-year revenue growth rates
    growth_rates = [(revenues[i] / revenues[i-1]) - 1 for i in range(1, n)]

    # EBIT margin = EBIT / Revenue (measures operational profitability)
    ebit_margins = [ebits[i] / revenues[i] for i in range(n)]

    # CapEx as % of revenue (capital intensity)
    capex_pct = [capex[i] / revenues[i] for i in range(n)]

    # D&A as % of revenue
    dep_pct = [depreciation[i] / revenues[i] for i in range(n)]

    return {
        "growth_rates"    : growth_rates,
        "avg_growth"      : statistics.mean(growth_rates),
        "ebit_margins"    : ebit_margins,
        "avg_ebit_margin" : statistics.mean(ebit_margins),
        "capex_pct"       : capex_pct,
        "avg_capex_pct"   : statistics.mean(capex_pct),
        "dep_pct"         : dep_pct,
        "avg_dep_pct"     : statistics.mean(dep_pct),
        "last_revenue"    : revenues[-1],
        "last_ebit"       : ebits[-1],
    }


# ==============================================================================
# SECTION 5: 5-YEAR FREE CASH FLOW PROJECTION
#
# Free Cash Flow (FCF) = the actual cash a business generates for investors,
# after paying for operations and capital expenditures.
#
# FCF = NOPAT + D&A − CapEx − ΔNWC
#
#   NOPAT  = Net Operating Profit After Tax = EBIT × (1 – Tax Rate)
#            This is what the company earns from operations, tax-adjusted.
#   D&A    = Depreciation & Amortisation (non-cash, so we add it back)
#   CapEx  = Capital Expenditure (cash spent on assets, so we deduct it)
#   ΔNWC   = Change in Net Working Capital (cash tied up in operations)
# ==============================================================================

def project_fcf(historical: dict) -> list:
    """
    Project Free Cash Flow for each of the next N years.

    We apply a growth rate to revenue, then estimate FCF using average
    historical margins and capital efficiency ratios.
    """
    n              = CONFIG["forecast_years"]
    tax            = CONFIG["tax_rate"]
    nwc_pct        = CONFIG["nwc_pct_revenue"]
    last_rev       = historical["last_revenue"]

    # Conservative growth: cap at max_growth_rate to avoid overoptimism
    proj_growth = min(historical["avg_growth"], CONFIG["max_growth_rate"])

    projections = []
    for yr in range(1, n + 1):
        # Revenue compounds at projected growth rate
        rev   = last_rev * ((1 + proj_growth) ** yr)

        # EBIT based on average historical EBIT margin
        ebit  = rev * historical["avg_ebit_margin"]

        # NOPAT: EBIT after taxes (interest is excluded since we're computing
        # enterprise-level FCF, capital-structure neutral)
        nopat = ebit * (1 - tax)

        # D&A (non-cash, added back — it's a paper charge, not real outflow)
        dep   = rev * historical["avg_dep_pct"]

        # CapEx (real cash spent on assets — subtract it)
        capex = rev * historical["avg_capex_pct"]

        # Change in NWC (cash consumed by working capital growth)
        d_nwc = rev * nwc_pct

        # Free Cash Flow to Firm (FCFF)
        fcf   = nopat + dep - capex - d_nwc

        projections.append({
            "year"        : yr,
            "revenue"     : rev,
            "ebit"        : ebit,
            "ebit_margin" : ebit / rev,
            "nopat"       : nopat,
            "depreciation": dep,
            "capex"       : capex,
            "delta_nwc"   : d_nwc,
            "fcf"         : fcf,
            "growth_rate" : proj_growth,
        })

    return projections


# ==============================================================================
# SECTION 6: TERMINAL VALUE & INTRINSIC PRICE
#
# Terminal Value captures all cash flows BEYOND Year 5 (in perpetuity).
# We use the Gordon Growth Model:
#
#   TV = FCF₅ × (1 + g) / (WACC − g)
#
#   g  = long-run terminal growth rate (≤ nominal GDP growth)
#
# This value is then discounted back to today (present value).
#
# Enterprise Value = PV of FCFs + PV of Terminal Value
# Equity Value     = Enterprise Value − Net Debt
# Intrinsic Price  = Equity Value / Shares Outstanding
# ==============================================================================

def calculate_intrinsic_value(projections: list, wacc_result: dict, data: dict) -> dict:
    """
    Discount FCFs and Terminal Value to derive the implied stock price.

    Finance Note
    ------------
    Terminal value often accounts for 60–80% of total enterprise value
    in stable, mature companies. This makes the terminal growth assumption
    extremely sensitive — always run a sensitivity analysis in practice.
    """
    wacc = wacc_result["wacc"]
    tg   = CONFIG["terminal_growth"]

    # Discount each year's FCF to present value
    pv_fcfs = []
    for p in projections:
        yr  = p["year"]
        pv  = p["fcf"] / ((1 + wacc) ** yr)
        pv_fcfs.append(pv)

    # Terminal Value (Gordon Growth Model) — using Year 5 FCF
    last_fcf     = projections[-1]["fcf"]
    terminal_val = last_fcf * (1 + tg) / (wacc - tg)

    # Discount terminal value to present
    n = CONFIG["forecast_years"]
    pv_terminal  = terminal_val / ((1 + wacc) ** n)

    # Enterprise Value = sum of all PV FCFs + PV of Terminal Value
    sum_pv_fcfs  = sum(pv_fcfs)
    enterprise_value = sum_pv_fcfs + pv_terminal

    # Equity Value: subtract net debt (Debt − Cash)
    net_debt     = data["total_debt"] - data["total_cash"]
    equity_value = enterprise_value - net_debt

    # Intrinsic Price per Share
    # Note: if data is in INR Crores, convert: 1 crore = 10,000,000
    # Equity value in crores × 10,000,000 / shares = price per share in INR
    if data["currency"] == "INR Crores":
        equity_value_inr = equity_value * 1e7          # crores → rupees
        intrinsic_price  = equity_value_inr / data["shares"]
    else:
        # If yfinance returns USD, shares are already in USD-equivalent basis
        intrinsic_price  = equity_value / data["shares"]

    current_price = data["current_price"]
    upside        = (intrinsic_price / current_price) - 1

    # Margin of Safety: how much the stock would need to fall for intrinsic
    # value to equal market price (Benjamin Graham's concept)
    margin_of_safety = 1 - (current_price / intrinsic_price) if intrinsic_price > 0 else None

    return {
        "pv_fcfs"          : pv_fcfs,
        "sum_pv_fcfs"      : sum_pv_fcfs,
        "terminal_value"   : terminal_val,
        "pv_terminal"      : pv_terminal,
        "tv_pct_of_ev"     : pv_terminal / enterprise_value,
        "enterprise_value" : enterprise_value,
        "net_debt"         : net_debt,
        "equity_value"     : equity_value,
        "intrinsic_price"  : intrinsic_price,
        "current_price"    : current_price,
        "upside_pct"       : upside,
        "margin_of_safety" : margin_of_safety,
    }


# ==============================================================================
# SECTION 7: SENSITIVITY ANALYSIS
# Real analysts always test how the output changes with key assumptions.
# This is a 2-variable sensitivity table: WACC × Terminal Growth Rate.
# ==============================================================================

def sensitivity_analysis(projections: list, data: dict, base_wacc: float) -> dict:
    """
    Compute a grid of intrinsic prices across different WACC and terminal
    growth rate assumptions.

    This is the most important slide in any real DCF — it shows the *range*
    of outcomes, not a single misleadingly precise number.
    """
    wacc_range = [base_wacc - 0.02, base_wacc - 0.01, base_wacc,
                  base_wacc + 0.01, base_wacc + 0.02]
    tg_range   = [0.02, 0.03, 0.04, 0.05, 0.06]

    last_fcf = projections[-1]["fcf"]
    n        = CONFIG["forecast_years"]

    grid = {}
    for w in wacc_range:
        row = {}
        for g in tg_range:
            pv_fcfs_total = sum(
                projections[i]["fcf"] / ((1 + w) ** (i + 1))
                for i in range(n)
            )
            tv     = last_fcf * (1 + g) / (w - g) if w > g else float("nan")
            pv_tv  = tv / ((1 + w) ** n)
            ev     = pv_fcfs_total + pv_tv
            net_d  = data["total_debt"] - data["total_cash"]
            eq_val = ev - net_d

            if data["currency"] == "INR Crores":
                price = (eq_val * 1e7) / data["shares"]
            else:
                price = eq_val / data["shares"]

            row[g] = price
        grid[w] = row

    return {"wacc_range": wacc_range, "tg_range": tg_range, "grid": grid}


# ==============================================================================
# SECTION 8: OUTPUT / DISPLAY
# Formatted console output for a clean portfolio presentation.
# ==============================================================================

def _fmt(val, unit="", decimals=1):
    """Helper: format large numbers with commas."""
    if isinstance(val, float) and (val != val):  # NaN check
        return " N/A "
    if unit == "%":
        return f"{val * 100:.{decimals}f}%"
    if unit in ("cr", "₹"):
        return f"₹ {val:,.{decimals}f}"
    return f"{val:,.{decimals}f}"


def print_banner():
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║          DCF VALUATION MODEL — {CONFIG['company_name']:<38}║
║          Ticker: {CONFIG['ticker']:<10}  |  Date: {datetime.today().strftime('%d %b %Y'):<28}║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_wacc(wacc_result: dict):
    print("━" * 72)
    print("  SECTION A: WACC CALCULATION")
    print("━" * 72)
    rows = [
        ["Risk-Free Rate (10-yr G-Sec)", _fmt(wacc_result["risk_free_rate"], "%")],
        ["Beta (5-yr monthly)",          f"{wacc_result['beta']:.2f}"],
        ["Equity Risk Premium",          _fmt(wacc_result["erp"], "%")],
        ["Cost of Equity (CAPM)",        _fmt(wacc_result["cost_of_equity"], "%")],
        ["Cost of Debt (pre-tax)",       _fmt(wacc_result["cost_of_debt"], "%")],
        ["Tax Rate",                     _fmt(CONFIG["tax_rate"], "%")],
        ["After-Tax Cost of Debt",       _fmt(wacc_result["tax_shield"], "%")],
        ["Weight — Equity",              _fmt(wacc_result["weight_equity"], "%")],
        ["Weight — Debt",                _fmt(wacc_result["weight_debt"], "%")],
        ["",                             ""],
        ["★  WACC",                      _fmt(wacc_result["wacc"], "%", 2)],
    ]
    for label, value in rows:
        print(f"  {label:<40} {value}")
    print()


def print_fcf_table(projections: list, results: dict):
    print("━" * 72)
    print("  SECTION B: 5-YEAR FREE CASH FLOW PROJECTIONS  (INR Crores)")
    print("━" * 72)

    header = f"  {'Metric':<22}" + "".join(f"{'FY' + str(2025 + p['year']-1):>10}" for p in projections)
    print(header)
    print("  " + "-" * 68)

    def row(label, key, pct=False, cr=True):
        vals = [p[key] for p in projections]
        line = f"  {label:<22}"
        for v in vals:
            if pct:
                line += f"  {v*100:>6.1f}%  "
            elif cr:
                line += f"{v:>10,.0f}"
            else:
                line += f"{v:>10,.2f}"
        print(line)

    row("Revenue",            "revenue")
    row("EBIT",               "ebit")
    row("  EBIT Margin",      "ebit_margin", pct=True)
    row("NOPAT",              "nopat")
    row("  + D&A",            "depreciation")
    row("  − CapEx",          "capex")
    row("  − ΔNWC",           "delta_nwc")
    print("  " + "-" * 68)
    row("Free Cash Flow",     "fcf")
    print()

    # PV of FCFs
    print(f"  {'PV of FCFs':<22}", end="")
    for pv in results["pv_fcfs"]:
        print(f"{pv:>10,.0f}", end="")
    print()
    print()


def print_valuation_summary(results: dict, data: dict, wacc_result: dict):
    print("━" * 72)
    print("  SECTION C: VALUATION BRIDGE")
    print("━" * 72)
    ccy = data["currency"]

    rows = [
        ["Sum of PV FCFs",        f"{results['sum_pv_fcfs']:>12,.0f}  {ccy}"],
        ["Terminal Value (Y5)",   f"{results['terminal_value']:>12,.0f}  {ccy}"],
        ["PV of Terminal Value",  f"{results['pv_terminal']:>12,.0f}  {ccy}"],
        [" % of Enterprise Value",f"{results['tv_pct_of_ev']*100:>12.1f}%"],
        ["",                      ""],
        ["Enterprise Value",      f"{results['enterprise_value']:>12,.0f}  {ccy}"],
        ["  − Net Debt",          f"{results['net_debt']:>12,.0f}  {ccy}"],
        ["",                      ""],
        ["Equity Value",          f"{results['equity_value']:>12,.0f}  {ccy}"],
    ]
    for label, value in rows:
        print(f"  {label:<32} {value}")
    print()


def print_final_verdict(results: dict):
    print("━" * 72)
    print("  SECTION D: INTRINSIC VALUE vs MARKET PRICE")
    print("━" * 72)

    iv   = results["intrinsic_price"]
    cp   = results["current_price"]
    up   = results["upside_pct"]
    mos  = results["margin_of_safety"]
    sign = "▲" if up > 0 else "▼"

    print(f"  Intrinsic Value (DCF)  :  ₹ {iv:>8,.2f}")
    print(f"  Current Market Price   :  ₹ {cp:>8,.2f}")
    print(f"  Implied Upside/Down    :  {sign} {abs(up)*100:>5.1f}%")
    if mos is not None:
        print(f"  Margin of Safety       :  {mos*100:>5.1f}%")
    print()

    if up > 0.15:
        verdict = "🟢  UNDERVALUED  — Market price appears below intrinsic value."
    elif up < -0.15:
        verdict = "🔴  OVERVALUED   — Market price appears above intrinsic value."
    else:
        verdict = "🟡  FAIRLY VALUED — Price is within ±15% of intrinsic value."

    print(f"  Verdict: {verdict}")
    print()

    # Important caveat (every real analyst writes one)
    print("  ⚠️  ANALYST NOTE")
    print("  This DCF is a mechanistic model using historical averages.")
    print("  It does NOT account for: AI-driven demand tailwinds, deal wins,")
    print("  management quality, currency risk (USD/INR), or macro shocks.")
    print("  Always triangulate with comparable company multiples (EV/EBITDA,")
    print("  P/E) and precedent transaction analysis.")
    print()


def print_sensitivity(sens: dict, data: dict):
    print("━" * 72)
    print("  SECTION E: SENSITIVITY ANALYSIS — Intrinsic Price (₹)")
    print("  Rows = WACC  |  Columns = Terminal Growth Rate")
    print("━" * 72)

    tg_range   = sens["tg_range"]
    wacc_range = sens["wacc_range"]
    grid       = sens["grid"]

    # Header row
    header = f"  {'WACC \\ TGR':>12}"
    for g in tg_range:
        header += f"  {g*100:.0f}%    "
    print(header)
    print("  " + "-" * 66)

    for w in wacc_range:
        line = f"  {w*100:>10.1f}%"
        for g in tg_range:
            price = grid[w][g]
            if price != price:  # NaN
                line += f"{'N/A':>8}"
            else:
                line += f"  {price:>7,.0f}"
        print(line)
    print()
    print(f"  Current Market Price: ₹ {data['current_price']:,.0f}")
    print()


# ==============================================================================
# SECTION 9: MAIN EXECUTION
# ==============================================================================

def main():
    print_banner()

    # 1. Load data
    data = load_data()
    print(f"  Company    : {CONFIG['company_name']}")
    print(f"  Data Source: {data['source']}")
    print(f"  Currency   : {data['currency']}")
    print(f"  Share Price: ₹ {data['current_price']:,.2f}")
    print(f"  Market Cap : {data['currency']} {data['market_cap']:,.0f}\n")

    # 2. WACC
    wacc_result  = calculate_wacc(data)
    print_wacc(wacc_result)

    # 3. Historical analysis
    historical   = analyse_historical(data)
    print(f"  Historical avg revenue growth : {historical['avg_growth']*100:.1f}%")
    print(f"  Used projection growth rate   : {min(historical['avg_growth'], CONFIG['max_growth_rate'])*100:.1f}%")
    print(f"  Historical avg EBIT margin    : {historical['avg_ebit_margin']*100:.1f}%")
    print(f"  Terminal growth rate          : {CONFIG['terminal_growth']*100:.1f}%\n")

    # 4. FCF projections
    projections  = project_fcf(historical)

    # 5. Intrinsic value
    results      = calculate_intrinsic_value(projections, wacc_result, data)

    # 6. Print tables
    print_fcf_table(projections, results)
    print_valuation_summary(results, data, wacc_result)
    print_final_verdict(results)

    # 7. Sensitivity
    sens = sensitivity_analysis(projections, data, wacc_result["wacc"])
    print_sensitivity(sens, data)

    print("━" * 72)
    print("  Model complete. Built for GitHub Portfolio | Infosys DCF Valuation")
    print("━" * 72)


if __name__ == "__main__":
    main()
