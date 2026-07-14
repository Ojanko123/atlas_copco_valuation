"""
atlas_corpo_project.py

Valuation toolkit for Atlas Copco AB (Nasdaq Stockholm: ATCO B), FY2025
figures where available.

Models:
  1. CAPM-built WACC          - cost of equity/debt from first principles,
                                 not a hardcoded number.
  2. ROIC & economic profit   - the piece that actually explains why a
                                 quality compounder trades above every
                                 "floor value" method below. Also the
                                 source of a fundamentally-derived growth
                                 rate, used to keep the DCF and LBO models
                                 telling the same economic story instead
                                 of contradicting each other.
  3. Reverse DCF (+ Monte Carlo) - what growth rate does the market price
                                 imply, and how sensitive is that to WACC/
                                 terminal growth uncertainty.
  4. Fundamental DCF          - a real forward valuation using the ROIC-
                                 derived growth rate, so there's an actual
                                 DCF price to compare against the market,
                                 not just an implied growth rate.
  5. Sum of the Parts         - value each segment on its own multiple.
  6. Ability to Pay (LBO)     - max price a leveraged financial buyer
                                 could justify at a target IRR, using the
                                 same ROIC-derived growth rate as (4).
  7. EPV                      - Greenwald no-growth floor value, plus a
                                 real reproduction-value estimate (R&D and
                                 brand/SG&A capitalized as intangible
                                 assets, not an arbitrary multiplier).

Reported (real, FY2025 annual report / Q4 report): revenue, adjusted
EBITA, net income, shares outstanding, share price, dividend. Shares
outstanding is derived from Investor AB's disclosed stake (840,053,755
shares = 17.1% of capital).

Estimated (flagged inline): segment splits, net debt, depreciation, SG&A,
PP&E, net working capital, CAPM inputs, useful lives. Replace with real
statement data if this needs to be more than a teaching model.

IMPORTANT modeling note: Atlas Copco reports adjusted EBITA (EBIT before
amortization of acquisition-related intangibles), not EBITDA. EBITA still
includes depreciation. Peer trading multiples for LBO/leverage purposes
are conventionally EV/EBITDA, so this file explicitly reconciles EBITA to
EBITDA via an estimated depreciation add-back rather than using the two
terms interchangeably.

Run:
    pip install numpy scipy matplotlib
    python atlas_corpo_project.py
"""

from dataclasses import dataclass, field
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# ---------------------------------------------------------------------
# Company data
# ---------------------------------------------------------------------

@dataclass
class Segment:
    name: str
    revenue_msek: float
    ebit_margin: float
    peer_ev_ebit_multiple: float

    @property
    def ebit_msek(self) -> float:
        return self.revenue_msek * self.ebit_margin


@dataclass
class CompanyData:
    company_name: str = "Atlas Copco AB"
    ticker: str = "ATCO B"
    fiscal_year: int = 2025

    # --- reported ---
    revenue_msek: float = 168_340.0
    adjusted_ebita_msek: float = 34_914.0     # EBITA, not EBITDA - see module docstring
    adjusted_ebita_margin: float = 0.207
    net_income_msek: float = 26_420.0
    shares_outstanding_m: float = 4_912.0
    share_price_sek: float = 166.0
    dividend_per_share_sek: float = 5.00

    # --- estimated: balance sheet & tax ---
    net_debt_msek: float = 15_000.0
    tax_rate: float = 0.22
    net_ppe_msek: float = 35_000.0
    net_working_capital_msek: float = 20_000.0
    depreciation_pct_of_revenue: float = 0.03   # to reconcile EBITA -> EBITDA

    # --- estimated: CAPM inputs (WACC is built, not assumed - see compute_wacc_capm) ---
    risk_free_rate: float = 0.025      # ~Swedish/German 10Y government bond
    levered_beta: float = 1.10         # industrial capital-goods peer range
    equity_risk_premium: float = 0.050
    credit_spread: float = 0.015       # over risk-free, for an A/AA-ish industrial

    terminal_growth_rate: float = 0.025

    # --- estimated: intangible-capitalization inputs (feed EPV reproduction value AND invested capital/ROIC) ---
    sga_pct_of_revenue: float = 0.12
    capitalizable_sga_fraction: float = 0.5
    rd_useful_life_years: int = 6
    sga_useful_life_years: int = 4
    historical_spend_growth_rate: float = 0.03

    segments: list = field(default_factory=lambda: [
        Segment("Compressor Technique", 168_340 * 0.51, 0.25, 17.0),
        Segment("Vacuum Technique", 168_340 * 0.19, 0.24, 19.0),
        Segment("Industrial Technique", 168_340 * 0.10, 0.19, 14.0),
        Segment("Power Technique", 168_340 * 0.20, 0.17, 13.0),
    ])

    # populated after CAPM WACC is computed, in main() - see compute_wacc_capm
    wacc: float = None
    cost_of_debt_pre_tax: float = None

    @property
    def market_cap_msek(self) -> float:
        return self.shares_outstanding_m * self.share_price_sek

    @property
    def enterprise_value_msek(self) -> float:
        return self.market_cap_msek + self.net_debt_msek

    @property
    def rd_spend_msek(self) -> float:
        return self.revenue_msek * 0.04  # Atlas Copco: ~4% of revenue (annual report)

    @property
    def sga_spend_msek(self) -> float:
        return self.revenue_msek * self.sga_pct_of_revenue

    @property
    def depreciation_msek(self) -> float:
        return self.revenue_msek * self.depreciation_pct_of_revenue

    @property
    def adjusted_ebitda_msek(self) -> float:
        """EBITA + depreciation. Use this, not adjusted_ebita_msek, for any
        EV/EBITDA multiple or leverage calculation."""
        return self.adjusted_ebita_msek + self.depreciation_msek


# ---------------------------------------------------------------------
# 0. WACC - built from CAPM, not assumed
# ---------------------------------------------------------------------

def compute_wacc_capm(company: CompanyData) -> dict:
    """
    Cost of equity:  Re = Rf + beta x ERP
    Cost of debt:    Rd = Rf + credit spread, after-tax = Rd x (1 - tax)
    Weights:         market-value E / (E+D), D / (E+D)
    WACC = E/V x Re + D/V x Rd(after-tax)

    Caveat worth stating out loud: the beta here is an assumed input, not
    fitted from return data. A more defensible build would use a bottom-up
    beta - unlever a set of industrial-automation peer betas, take the
    median, relever at Atlas Copco's own capital structure - rather than a
    single assumed levered beta. That's flagged as the natural next
    upgrade, not done here.
    """
    cost_of_equity = company.risk_free_rate + company.levered_beta * company.equity_risk_premium
    cost_of_debt_pretax = company.risk_free_rate + company.credit_spread
    cost_of_debt_aftertax = cost_of_debt_pretax * (1 - company.tax_rate)

    E = company.market_cap_msek
    D = company.net_debt_msek
    V = E + D
    e_weight = E / V
    d_weight = D / V

    wacc = e_weight * cost_of_equity + d_weight * cost_of_debt_aftertax

    return {
        "cost_of_equity": cost_of_equity, "cost_of_debt_pretax": cost_of_debt_pretax,
        "cost_of_debt_aftertax": cost_of_debt_aftertax,
        "e_weight": e_weight, "d_weight": d_weight, "wacc": wacc,
    }


def print_wacc(w: dict) -> None:
    print("WACC (CAPM build)")
    print(f"Cost of equity: {w['cost_of_equity']:.2%}  |  "
          f"Cost of debt (pre-tax {w['cost_of_debt_pretax']:.2%}, after-tax {w['cost_of_debt_aftertax']:.2%})")
    print(f"Weights: equity {w['e_weight']:.1%} / debt {w['d_weight']:.1%}  "
          f"(Atlas Copco is almost entirely equity-financed)")
    print(f"WACC: {w['wacc']:.2%}")


# ---------------------------------------------------------------------
# 0b. ROIC & economic profit - the piece that explains the valuation gap,
#     and the source of a growth rate used consistently in DCF and LBO
# ---------------------------------------------------------------------

def compute_roic_and_economic_profit(company: CompanyData, invested_capital_msek: float) -> dict:
    """
    ROIC = NOPAT / invested capital
    Economic profit = invested capital x (ROIC - WACC)
                     = NOPAT - invested capital x WACC   (equivalent, EVA-style)

    invested_capital_msek is passed in rather than recomputed here, because
    it's the same "reproduction value of operating assets" figure used in
    the EPV section (PP&E + NWC + capitalized R&D + capitalized brand
    spend) - one number, two uses, instead of inventing a second one.
    """
    nopat = company.adjusted_ebita_msek * (1 - company.tax_rate)
    roic = nopat / invested_capital_msek
    economic_spread = roic - company.wacc
    economic_profit_msek = invested_capital_msek * economic_spread

    return {
        "nopat_msek": nopat, "invested_capital_msek": invested_capital_msek,
        "roic": roic, "wacc": company.wacc, "economic_spread": economic_spread,
        "economic_profit_msek": economic_profit_msek,
    }


def estimate_fundamental_growth(roic_result: dict, fcf_conversion: float = 0.82) -> dict:
    """
    Sustainable growth = ROIC x reinvestment rate. Reinvestment rate is
    read off the same fcf_conversion assumption used in the DCF (the
    share of NOPAT that does NOT become free cash flow is, by
    construction, what's being reinvested).

    Using this one growth rate as the default for BOTH the fundamental DCF
    and the LBO's EBITDA growth assumption is the fix for the biggest
    structural issue in the previous version: a DCF implying ~20%+ growth
    coexisting with an LBO model assuming 5% growth, with nothing
    connecting them. They should tell the same economic story unless
    there's a specific reason (e.g. financial engineering vs. operating
    performance) for them to diverge - and if they still diverge, that
    divergence becomes the interesting finding rather than a modeling bug.
    """
    reinvestment_rate = 1 - fcf_conversion
    growth = roic_result["roic"] * reinvestment_rate
    return {"reinvestment_rate": reinvestment_rate, "fundamental_growth_rate": growth}


def print_roic(r: dict, g: dict) -> None:
    print("ROIC & ECONOMIC PROFIT")
    print(f"NOPAT: SEK {r['nopat_msek']:,.0f}m  |  Invested capital (reproduction-value proxy): "
          f"SEK {r['invested_capital_msek']:,.0f}m")
    print(f"ROIC: {r['roic']:.1%}  |  WACC: {r['wacc']:.1%}  |  "
          f"Spread: {r['economic_spread']:+.1%}")
    print(f"Economic profit: SEK {r['economic_profit_msek']:,.0f}m/year "
          f"(value created above the cost of capital)")
    print(f"Reinvestment rate: {g['reinvestment_rate']:.0%}  ->  "
          f"fundamentally-derived sustainable growth: {g['fundamental_growth_rate']:.2%}")
    print("This growth rate is what's used as the default assumption in both")
    print("the fundamental DCF and the LBO model below, so they're not")
    print("contradicting each other on how fast the business actually grows.")


def plot_roic(r: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    labels = ["ROIC", "WACC"]
    values = [r["roic"] * 100, r["wacc"] * 100]
    colors = ["#55A868", "#C44E52"]
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.4, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("%")
    ax.set_title(f"ROIC vs WACC\nspread {r['economic_spread']:+.1%}  |  "
                 f"economic profit SEK {r['economic_profit_msek']:,.0f}m/yr")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# 1. Reverse DCF - solve for the growth rate the market price implies
# ---------------------------------------------------------------------

def unlevered_fcf0(company: CompanyData, fcf_conversion: float = 0.82) -> float:
    nopat = company.adjusted_ebita_msek * (1 - company.tax_rate)
    return nopat * fcf_conversion


def dcf_value(fcf0, growth_rate, years, wacc, terminal_growth) -> float:
    if wacc <= terminal_growth:
        raise ValueError("WACC must exceed terminal growth rate.")
    pv_explicit = 0.0
    fcf = fcf0
    for t in range(1, years + 1):
        fcf *= (1 + growth_rate)
        pv_explicit += fcf / (1 + wacc) ** t
    terminal_value = fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    return pv_explicit + terminal_value / (1 + wacc) ** years


def solve_implied_growth(company: CompanyData, years: int = 5, fcf_conversion: float = 0.82) -> dict:
    fcf0 = unlevered_fcf0(company, fcf_conversion)
    target_ev = company.enterprise_value_msek
    wacc = company.wacc
    tg = company.terminal_growth_rate

    g_star = brentq(lambda g: dcf_value(fcf0, g, years, wacc, tg) - target_ev,
                     -0.20, 0.40, xtol=1e-6)

    sensitivity = {}
    for w in (wacc - 0.01, wacc, wacc + 0.01):
        for tgr in (tg - 0.005, tg, tg + 0.005):
            try:
                sensitivity[(round(w, 4), round(tgr, 4))] = brentq(
                    lambda g, w=w, tgr=tgr: dcf_value(fcf0, g, years, w, tgr) - target_ev,
                    -0.30, 0.60, xtol=1e-6)
            except ValueError:
                sensitivity[(round(w, 4), round(tgr, 4))] = float("nan")

    return {
        "implied_growth_rate": g_star, "years": years, "wacc": wacc,
        "terminal_growth_rate": tg, "fcf0_msek": fcf0,
        "target_ev_msek": target_ev, "sensitivity": sensitivity,
        "fcf_conversion": fcf_conversion,
    }


def print_reverse_dcf(r: dict, fundamental_growth: float) -> None:
    print("REVERSE DCF (diagnostic: what does the market price imply?)")
    print(f"Target EV: SEK {r['target_ev_msek']:,.0f}m | FCF0: SEK {r['fcf0_msek']:,.0f}m | "
          f"WACC: {r['wacc']:.1%} | Terminal g: {r['terminal_growth_rate']:.1%} | "
          f"Horizon: {r['years']}y")
    print(f"Implied 5-year FCF growth rate: {r['implied_growth_rate']:.2%}")
    print(f"Fundamentally-derived growth rate (ROIC x reinvestment): {fundamental_growth:.2%}")
    gap = r["implied_growth_rate"] - fundamental_growth
    print(f"Gap: {gap:+.1%} pts")
    print()
    tgs = sorted({k[1] for k in r["sensitivity"]})
    print(f"{'WACC \\ term g':>14}", end="")
    for tgr in tgs:
        print(f"{tgr:>9.2%}", end="")
    print()
    for w in sorted({k[0] for k in r["sensitivity"]}):
        print(f"{w:>14.2%}", end="")
        for tgr in tgs:
            print(f"{r['sensitivity'][(w, tgr)]:>9.2%}", end="")
        print()


# ---------------------------------------------------------------------
# 1b. Fundamental DCF - an actual forward valuation, using the ROIC-
#     derived growth rate. This is the number the reverse DCF alone
#     doesn't give you: a price to compare against the market price.
# ---------------------------------------------------------------------

def run_fundamental_dcf(company: CompanyData, growth_rate: float, years: int = 5,
                         fcf_conversion: float = 0.82) -> dict:
    fcf0 = unlevered_fcf0(company, fcf_conversion)
    ev = dcf_value(fcf0, growth_rate, years, company.wacc, company.terminal_growth_rate)
    equity_value = ev - company.net_debt_msek
    price_per_share = equity_value / company.shares_outstanding_m
    return {
        "growth_rate": growth_rate, "fcf0_msek": fcf0, "ev_msek": ev,
        "equity_value_msek": equity_value, "price_per_share_sek": price_per_share,
    }


def print_fundamental_dcf(r: dict, company: CompanyData) -> None:
    print("FUNDAMENTAL DCF (base case, ROIC-derived growth)")
    print(f"Growth assumption: {r['growth_rate']:.2%} | EV: SEK {r['ev_msek']:,.0f}m")
    print(f"Fundamental DCF value/share: SEK {r['price_per_share_sek']:,.2f}  "
          f"(market: SEK {company.share_price_sek:,.2f}, "
          f"{(r['price_per_share_sek'] - company.share_price_sek) / company.share_price_sek:+.1%})")


# ---------------------------------------------------------------------
# 1c. Monte Carlo on the reverse DCF
# ---------------------------------------------------------------------

def monte_carlo_reverse_dcf(company: CompanyData, years: int = 5,
                             n_sims: int = 5000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    wacc_samples = rng.normal(company.wacc, 0.010, n_sims)
    tg_samples = rng.normal(company.terminal_growth_rate, 0.005, n_sims)
    fcf_conv_samples = rng.uniform(0.75, 0.90, n_sims)

    target_ev = company.enterprise_value_msek
    implied = []
    for w, tg, fc in zip(wacc_samples, tg_samples, fcf_conv_samples):
        if w <= tg + 0.005:
            continue
        fcf0 = company.adjusted_ebita_msek * (1 - company.tax_rate) * fc
        try:
            g = brentq(lambda g: dcf_value(fcf0, g, years, w, tg) - target_ev,
                       -0.5, 0.9, xtol=1e-5)
            implied.append(g)
        except ValueError:
            continue

    arr = np.array(implied)
    return {
        "n_valid": len(arr), "n_requested": n_sims,
        "mean": float(np.mean(arr)), "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)), "p95": float(np.percentile(arr, 95)),
        "samples": arr,
    }


def print_monte_carlo(r: dict) -> None:
    print("REVERSE DCF - MONTE CARLO")
    print(f"Valid draws: {r['n_valid']}/{r['n_requested']}")
    print(f"Implied growth: mean {r['mean']:.2%} | median {r['median']:.2%} | "
          f"std {r['std']:.2%} | 5th-95th pct [{r['p5']:.2%}, {r['p95']:.2%}]")


def plot_monte_carlo(r: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(r["samples"] * 100, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.axvline(r["median"] * 100, color="#333333", linestyle="--", linewidth=1,
               label=f"median {r['median']:.1%}")
    ax.set_xlabel("Implied growth rate (%)")
    ax.set_ylabel("Draws")
    ax.set_title("Reverse DCF - distribution of implied growth\nunder WACC / terminal growth uncertainty")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# 2. Sum of the Parts - value each business area on its own multiple
#    (EV/EBIT, deliberately - segment-level D&A isn't estimated here,
#    so EBITDA multiples at the segment level would be fabricated)
# ---------------------------------------------------------------------

def run_sotp(company: CompanyData) -> dict:
    segment_valuations = []
    total_ev = 0.0
    for seg in company.segments:
        ev = seg.ebit_msek * seg.peer_ev_ebit_multiple
        segment_valuations.append({"segment": seg, "ebit_msek": seg.ebit_msek, "ev_msek": ev})
        total_ev += ev

    equity_value = total_ev - company.net_debt_msek
    value_per_share = equity_value / company.shares_outstanding_m
    premium = (value_per_share - company.share_price_sek) / company.share_price_sek

    return {
        "segment_valuations": segment_valuations, "total_ev_msek": total_ev,
        "net_debt_msek": company.net_debt_msek, "equity_value_msek": equity_value,
        "value_per_share_sek": value_per_share, "current_price_sek": company.share_price_sek,
        "premium_discount_to_market": premium,
    }


def print_sotp(r: dict) -> None:
    print("SUM OF THE PARTS (EV/EBIT by segment)")
    print(f"{'Segment':<24}{'EBIT (m)':>12}{'Multiple':>10}{'EV (m)':>14}")
    for sv in r["segment_valuations"]:
        print(f"{sv['segment'].name:<24}{sv['ebit_msek']:>12,.0f}"
              f"{sv['segment'].peer_ev_ebit_multiple:>9.1f}x{sv['ev_msek']:>14,.0f}")
    print(f"{'Total EV':<24}{'':>12}{'':>10}{r['total_ev_msek']:>14,.0f}")
    print(f"{'Less net debt':<24}{'':>12}{'':>10}{-r['net_debt_msek']:>14,.0f}")
    print(f"{'Implied equity value':<24}{'':>12}{'':>10}{r['equity_value_msek']:>14,.0f}")
    print()
    print(f"SOTP value/share: SEK {r['value_per_share_sek']:,.2f}  "
          f"(market: SEK {r['current_price_sek']:,.2f}, "
          f"{r['premium_discount_to_market']:+.1%})")


def plot_sotp(r: dict, path: str) -> None:
    names = [sv["segment"].name for sv in r["segment_valuations"]]
    evs = [sv["ev_msek"] for sv in r["segment_valuations"]]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, evs, color=colors)
    for bar, val in zip(bars, evs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 5000, f"{val:,.0f}",
                 ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Segment EV (SEK m)")
    ax.set_title("Sum of the Parts: EV by segment")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# 3. Ability to Pay - max price a financial (LBO) buyer could justify.
#    This answers a DIFFERENT question than "what is Atlas Copco worth" -
#    see the explanatory note printed alongside it.
# ---------------------------------------------------------------------

def _simulate_deal(entry_multiple, exit_multiple, entry_ebitda, hold_years, leverage_x,
                    ebitda_growth, cost_of_debt, fcf_conversion, tax_rate) -> dict:
    entry_ev = entry_multiple * entry_ebitda
    entry_debt = leverage_x * entry_ebitda
    entry_equity = entry_ev - entry_debt

    debt = entry_debt
    ebitda = entry_ebitda
    for _ in range(hold_years):
        ebitda *= (1 + ebitda_growth)
        interest = debt * cost_of_debt
        fcf = ebitda * fcf_conversion * (1 - tax_rate) - interest
        debt = max(0.0, debt - max(0.0, fcf))

    exit_ev = exit_multiple * ebitda
    exit_debt = debt
    exit_equity = max(0.0, exit_ev - exit_debt)

    return {
        "entry_ev": entry_ev, "entry_debt": entry_debt, "entry_equity": entry_equity,
        "entry_ebitda": entry_ebitda, "exit_ebitda": ebitda,
        "exit_ev": exit_ev, "exit_debt": exit_debt, "exit_equity": exit_equity,
        "entry_multiple": entry_multiple, "exit_multiple": exit_multiple,
        "hold_years": hold_years,
    }


def _irr_from_deal(deal: dict) -> float:
    if deal["entry_equity"] <= 0:
        return float("inf")
    if deal["exit_equity"] <= 0:
        return -1.0
    return (deal["exit_equity"] / deal["entry_equity"]) ** (1 / deal["hold_years"]) - 1


def solve_max_entry_multiple(company: CompanyData, target_irr: float = 0.18,
                              hold_years: int = 5, leverage_x_ebitda: float = 3.5,
                              exit_multiple: float = None, ebitda_growth: float = None,
                              fcf_conversion: float = 0.75) -> dict:
    """ebitda_growth defaults to None -> caller should pass the same
    fundamentally-derived growth rate used in the DCF, so the two models
    aren't assuming different futures for the same company."""
    entry_ebitda = company.adjusted_ebitda_msek  # EBITDA, correctly, not EBITA
    exit_is_same_as_entry = exit_multiple is None

    def irr_gap(x):
        exit_mult = x if exit_is_same_as_entry else exit_multiple
        deal = _simulate_deal(x, exit_mult, entry_ebitda, hold_years, leverage_x_ebitda,
                               ebitda_growth, company.cost_of_debt_pre_tax, fcf_conversion,
                               company.tax_rate)
        return _irr_from_deal(deal) - target_irr

    max_multiple = brentq(irr_gap, 1.0, 40.0, xtol=1e-5)
    resolved_exit_multiple = max_multiple if exit_is_same_as_entry else exit_multiple

    deal = _simulate_deal(max_multiple, resolved_exit_multiple, entry_ebitda, hold_years,
                           leverage_x_ebitda, ebitda_growth, company.cost_of_debt_pre_tax,
                           fcf_conversion, company.tax_rate)

    equity_value_to_shareholders = deal["entry_ev"] - company.net_debt_msek
    max_price_per_share = equity_value_to_shareholders / company.shares_outstanding_m
    premium = (max_price_per_share - company.share_price_sek) / company.share_price_sek

    return {
        "target_irr": target_irr, "hold_years": hold_years,
        "leverage_x_ebitda": leverage_x_ebitda, "exit_multiple": resolved_exit_multiple,
        "ebitda_growth": ebitda_growth, "fcf_conversion": fcf_conversion,
        "max_entry_ev_ebitda_multiple": max_multiple, "deal": deal,
        "max_entry_ev_msek": deal["entry_ev"], "entry_debt_msek": deal["entry_debt"],
        "entry_equity_msek": deal["entry_equity"], "max_price_per_share_sek": max_price_per_share,
        "implied_premium_to_market": premium,
    }


def print_ability_to_pay(r: dict, company: CompanyData) -> None:
    print("ABILITY TO PAY (LBO breakeven price)")
    print("This is the ceiling a LEVERAGED FINANCIAL BUYER could pay and still")
    print(f"clear an {r['target_irr']:.0%} IRR over {r['hold_years']} years using {r['leverage_x_ebitda']:.1f}x debt -")
    print("not an estimate of what the company is worth. It's expected to sit")
    print("well below the market price for a high-quality, low-leverage compounder;")
    print("that gap is itself the finding (Atlas Copco isn't a natural LBO target).")
    print(f"EBITDA growth assumption used: {r['ebitda_growth']:.2%} (same as fundamental DCF)")
    print(f"Max entry multiple: {r['max_entry_ev_ebitda_multiple']:.1f}x EBITDA "
          f"(EV SEK {r['max_entry_ev_msek']:,.0f}m = debt {r['entry_debt_msek']:,.0f}m "
          f"+ sponsor equity {r['entry_equity_msek']:,.0f}m)")
    print(f"Max offer price/share: SEK {r['max_price_per_share_sek']:,.2f}  "
          f"(market: SEK {company.share_price_sek:,.2f}, "
          f"{r['implied_premium_to_market']:+.1%})")


def irr_bridge(deal: dict) -> dict:
    ev_growth_effect = deal["entry_multiple"] * (deal["exit_ebitda"] - deal["entry_ebitda"])
    ev_multiple_effect = deal["exit_ebitda"] * (deal["exit_multiple"] - deal["entry_multiple"])
    deleveraging_effect = deal["entry_debt"] - deal["exit_debt"]
    total_creation = deal["exit_equity"] - deal["entry_equity"]
    irr = _irr_from_deal(deal)

    def pct(x):
        return x / total_creation if total_creation != 0 else float("nan")

    return {
        "deal": deal, "irr": irr, "total_value_creation_msek": total_creation,
        "growth_effect_msek": ev_growth_effect, "multiple_effect_msek": ev_multiple_effect,
        "deleveraging_effect_msek": deleveraging_effect,
        "growth_pct": pct(ev_growth_effect), "multiple_pct": pct(ev_multiple_effect),
        "deleveraging_pct": pct(deleveraging_effect),
    }


def print_irr_bridge(b: dict, label: str) -> None:
    d = b["deal"]
    print(f"IRR BRIDGE - {label}")
    print(f"Entry {d['entry_multiple']:.1f}x -> Exit {d['exit_multiple']:.1f}x over "
          f"{d['hold_years']}y | IRR: {b['irr']:.1%}")
    print(f"  EBITDA growth:   SEK {b['growth_effect_msek']:>10,.0f}m  ({b['growth_pct']:>6.1%} of value created)")
    print(f"  Multiple change: SEK {b['multiple_effect_msek']:>10,.0f}m  ({b['multiple_pct']:>6.1%} of value created)")
    print(f"  Deleveraging:    SEK {b['deleveraging_effect_msek']:>10,.0f}m  ({b['deleveraging_pct']:>6.1%} of value created)")
    print(f"  Total equity value created: SEK {b['total_value_creation_msek']:,.0f}m")


def plot_irr_bridge(b: dict, path: str, title: str) -> None:
    """Waterfall: entry equity, then each effect stacked, then exit equity."""
    d = b["deal"]
    entry_equity = d["entry_equity"]
    growth = b["growth_effect_msek"]
    multiple = b["multiple_effect_msek"]
    delevering = b["deleveraging_effect_msek"]
    exit_equity = d["exit_equity"]

    labels = ["Entry\nequity", "EBITDA\ngrowth", "Multiple\nchange", "Deleveraging", "Exit\nequity"]
    heights = [entry_equity, growth, multiple, delevering, exit_equity]
    bottoms = [0, entry_equity, entry_equity + growth,
               entry_equity + growth + multiple, 0]
    colors = ["#333333", "#55A868", "#8172B2", "#4C72B0", "#333333"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, heights, bottom=bottoms, color=colors)
    ax.set_ylabel("SEK m")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# 4. EPV (Greenwald) - value assuming zero growth, forever. A FLOOR, not
#    a fair-value estimate - see explanatory note printed alongside it.
# ---------------------------------------------------------------------

def capitalize_spend_stream(current_annual_spend: float, historical_growth_rate: float,
                             useful_life_years: int) -> float:
    """Damodaran-style capitalization: treat recurring spend as building an
    asset amortized straight-line over useful_life_years. Reconstruct the
    trailing L years by deflating current spend at historical_growth_rate,
    sum each year's unamortized remainder."""
    L = useful_life_years
    total = 0.0
    for j in range(L):
        spend_j = current_annual_spend / (1 + historical_growth_rate) ** j
        remaining_fraction = (L - j) / L
        total += spend_j * remaining_fraction
    return total


def estimate_reproduction_value(company: CompanyData) -> dict:
    """Reproduction value of operating assets = book tangible assets
    (PP&E, NWC) + capitalized R&D + capitalized brand/customer-relationship
    spend (a slice of SG&A). Also used downstream as the invested-capital
    figure for ROIC."""
    capitalized_rd = capitalize_spend_stream(
        company.rd_spend_msek, company.historical_spend_growth_rate, company.rd_useful_life_years)
    capitalizable_sga = company.sga_spend_msek * company.capitalizable_sga_fraction
    capitalized_sga = capitalize_spend_stream(
        capitalizable_sga, company.historical_spend_growth_rate, company.sga_useful_life_years)

    total = company.net_ppe_msek + company.net_working_capital_msek + capitalized_rd + capitalized_sga

    return {
        "net_ppe_msek": company.net_ppe_msek,
        "net_working_capital_msek": company.net_working_capital_msek,
        "capitalized_rd_msek": capitalized_rd,
        "capitalized_sga_msek": capitalized_sga,
        "total_msek": total,
    }


def run_epv(company: CompanyData, reproduction: dict, normalization_haircut: float = 0.0) -> dict:
    normalized_ebit = company.adjusted_ebita_msek * (1 - normalization_haircut)
    nopat = normalized_ebit * (1 - company.tax_rate)
    epv_operations = nopat / company.wacc
    epv_equity = epv_operations - company.net_debt_msek
    epv_per_share = epv_equity / company.shares_outstanding_m

    repro_equity = reproduction["total_msek"] - company.net_debt_msek
    repro_per_share = repro_equity / company.shares_outstanding_m
    franchise_per_share = epv_per_share - repro_per_share

    return {
        "normalized_ebit_msek": normalized_ebit, "nopat_msek": nopat, "wacc": company.wacc,
        "epv_operations_msek": epv_operations, "epv_equity_msek": epv_equity,
        "epv_per_share_sek": epv_per_share, "current_price_sek": company.share_price_sek,
        "reproduction": reproduction,
        "reproduction_value_per_share_sek": repro_per_share,
        "franchise_value_per_share_sek": franchise_per_share,
    }


def print_epv(r: dict) -> None:
    print("EARNINGS POWER VALUE (zero-growth floor, not a fair-value estimate)")
    print(f"Normalized EBIT: SEK {r['normalized_ebit_msek']:,.0f}m | "
          f"NOPAT: SEK {r['nopat_msek']:,.0f}m | WACC: {r['wacc']:.1%}")
    print(f"EPV/share: SEK {r['epv_per_share_sek']:,.2f}  "
          f"(market: SEK {r['current_price_sek']:,.2f})")
    print()
    print("Reproduction value of operating assets:")
    rp = r["reproduction"]
    print(f"  Net PP&E (book):              SEK {rp['net_ppe_msek']:>10,.0f}m")
    print(f"  Net working capital (book):   SEK {rp['net_working_capital_msek']:>10,.0f}m")
    print(f"  Capitalized R&D:              SEK {rp['capitalized_rd_msek']:>10,.0f}m")
    print(f"  Capitalized brand/SG&A:       SEK {rp['capitalized_sga_msek']:>10,.0f}m")
    print(f"  Total reproduction value:     SEK {rp['total_msek']:>10,.0f}m")
    print(f"Reproduction value/share: SEK {r['reproduction_value_per_share_sek']:,.2f}")
    print(f"Implied franchise value/share:  SEK {r['franchise_value_per_share_sek']:,.2f}")


def plot_epv(r: dict, path: str) -> None:
    rp = r["reproduction"]
    labels = ["Net PP&E", "Net working\ncapital", "Capitalized\nR&D", "Capitalized\nbrand/SG&A"]
    values = [rp["net_ppe_msek"], rp["net_working_capital_msek"],
              rp["capitalized_rd_msek"], rp["capitalized_sga_msek"]]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylabel("SEK m")
    axes[0].set_title("Reproduction value components")
    axes[0].tick_params(axis="x", labelrotation=15)

    per_share_labels = ["EPV/share", "Reproduction\nvalue/share"]
    per_share_values = [r["epv_per_share_sek"], r["reproduction_value_per_share_sek"]]
    axes[1].bar(per_share_labels, per_share_values, color=["#4C72B0", "#333333"])
    axes[1].set_ylabel("SEK per share")
    axes[1].set_title(f"Franchise value/share: SEK {r['franchise_value_per_share_sek']:,.2f}")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    company = CompanyData()

    # WACC first - everything downstream depends on it
    wacc_result = compute_wacc_capm(company)
    company.wacc = wacc_result["wacc"]
    company.cost_of_debt_pre_tax = wacc_result["cost_of_debt_pretax"]

    print(f"{company.company_name} ({company.ticker}) - FY{company.fiscal_year}")
    print(f"Revenue SEK {company.revenue_msek:,.0f}m | "
          f"Adj. EBITA margin {company.adjusted_ebita_margin:.1%} | "
          f"Share price SEK {company.share_price_sek:,.2f} | "
          f"Market cap SEK {company.market_cap_msek:,.0f}m")
    print(f"Adj. EBITA: SEK {company.adjusted_ebita_msek:,.0f}m  ->  "
          f"Adj. EBITDA (+ estimated depreciation): SEK {company.adjusted_ebitda_msek:,.0f}m")
    print("-" * 70)

    print_wacc(wacc_result)
    print()

    reproduction = estimate_reproduction_value(company)
    roic_result = compute_roic_and_economic_profit(company, reproduction["total_msek"])
    fundamental_growth = estimate_fundamental_growth(roic_result, fcf_conversion=0.82)
    print_roic(roic_result, fundamental_growth)
    plot_roic(roic_result, "roic_vs_wacc.png")
    g_star = fundamental_growth["fundamental_growth_rate"]
    print()

    dcf_result = solve_implied_growth(company)
    print_reverse_dcf(dcf_result, g_star)
    print()

    fdcf_result = run_fundamental_dcf(company, growth_rate=g_star)
    print_fundamental_dcf(fdcf_result, company)
    print()

    mc_result = monte_carlo_reverse_dcf(company)
    print_monte_carlo(mc_result)
    plot_monte_carlo(mc_result, "reverse_dcf_monte_carlo.png")
    print()

    sotp_result = run_sotp(company)
    print_sotp(sotp_result)
    plot_sotp(sotp_result, "sotp_by_segment.png")
    print()

    atp_result = solve_max_entry_multiple(company, ebitda_growth=g_star)
    print_ability_to_pay(atp_result, company)
    print()

    breakeven_bridge = irr_bridge(atp_result["deal"])
    print_irr_bridge(breakeven_bridge, "breakeven deal (max price, no multiple expansion)")
    plot_irr_bridge(breakeven_bridge, "lbo_irr_bridge_breakeven.png",
                     "LBO value bridge - breakeven deal")
    print()

    illustrative_deal = _simulate_deal(
        entry_multiple=atp_result["max_entry_ev_ebitda_multiple"] * 0.75,
        exit_multiple=atp_result["max_entry_ev_ebitda_multiple"] * 0.75 + 1.0,
        entry_ebitda=company.adjusted_ebitda_msek, hold_years=atp_result["hold_years"],
        leverage_x=atp_result["leverage_x_ebitda"], ebitda_growth=atp_result["ebitda_growth"],
        cost_of_debt=company.cost_of_debt_pre_tax, fcf_conversion=atp_result["fcf_conversion"],
        tax_rate=company.tax_rate,
    )
    illustrative_bridge = irr_bridge(illustrative_deal)
    print_irr_bridge(illustrative_bridge, "illustrative deal (25% discount to max, +1x multiple expansion)")
    plot_irr_bridge(illustrative_bridge, "lbo_irr_bridge_illustrative.png",
                     "LBO value bridge - illustrative deal (25% discount, +1x exit)")
    print()

    epv_result = run_epv(company, reproduction)
    print_epv(epv_result)
    plot_epv(epv_result, "epv_reproduction_value.png")
    print()

    print("-" * 70)
    print("SUMMARY - implied value per share")
    values = {
        "Market price": company.share_price_sek,
        "EPV (no growth floor)": epv_result["epv_per_share_sek"],
        "Fundamental DCF": fdcf_result["price_per_share_sek"],
        "SOTP": sotp_result["value_per_share_sek"],
        "Max LBO price": atp_result["max_price_per_share_sek"],
    }
    for label, val in values.items():
        print(f"  {label:<22} SEK {val:,.2f}")
    print(f"  Reverse DCF implied growth (point estimate): {dcf_result['implied_growth_rate']:.2%}")
    print(f"  Reverse DCF implied growth (Monte Carlo median, 5-95pct): "
          f"{mc_result['median']:.2%} [{mc_result['p5']:.2%}, {mc_result['p95']:.2%}]")
    print(f"  Fundamental (ROIC-derived) growth used in DCF & LBO: {g_star:.2%}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(list(values.keys()), list(values.values()),
           color=["#333333", "#4C72B0", "#8172B2", "#55A868", "#C44E52"])
    ax.axhline(company.share_price_sek, color="#333333", linestyle="--", linewidth=1)
    ax.set_ylabel("SEK per share")
    ax.set_title(f"{company.company_name}: implied value per share")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("valuation_summary.png", dpi=150)
    plt.show()
    plt.close(fig)
    print("\nCharts saved: valuation_summary.png, reverse_dcf_monte_carlo.png,")
    print("sotp_by_segment.png, roic_vs_wacc.png, lbo_irr_bridge_breakeven.png,")
    print("lbo_irr_bridge_illustrative.png, epv_reproduction_value.png")


if __name__ == "__main__":
    main()
