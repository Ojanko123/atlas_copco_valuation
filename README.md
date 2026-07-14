# Atlas Copco AB Valuation
## Reverse DCF • Fundamental DCF • Sum-of-the-Parts • Leveraged Buyout • Earnings Power Value (Python)

> **Can Atlas Copco's premium market valuation be justified by fundamentals, or is the market pricing in near-perfect execution?**


# Overview

This project develops a comprehensive equity valuation framework for **Atlas Copco AB (Nasdaq Stockholm: ATCO B)** using publicly available financial statements and market data.

Rather than relying on a single valuation technique, the project combines multiple complementary valuation approaches used in investment banking, private equity, and equity research:

- Reverse Discounted Cash Flow (Reverse DCF)
- Fundamental Discounted Cash Flow (DCF)
- Sum-of-the-Parts (SOTP)
- Leveraged Buyout (LBO)
- Earnings Power Value (EPV)

The objective is not simply to estimate intrinsic value, but to understand what assumptions are already embedded in the market price and whether those expectations are supported by the company's underlying economics.

# Business Questions

The project addresses several practical valuation questions:

- What growth rate is currently implied by Atlas Copco's share price?
- Is that growth consistent with the company's operating fundamentals?
- What is the company's intrinsic value using a fundamentals-driven DCF?
- What is Atlas Copco worth using a Sum-of-the-Parts valuation?
- Could a private equity sponsor justify acquiring the business in a leveraged buyout?
- How much of Atlas Copco's value comes from sustainable competitive advantages rather than tangible assets?

# Company

**Atlas Copco AB (ATCO B)**

Industrial Equipment & Automation

Fiscal Year: **FY2025**

### Financial Highlights

| Metric | Value |
|---------|---------:|
| Revenue | SEK 168.3 bn |
| Adjusted EBITA | SEK 34.9 bn |
| EBITA Margin | 20.7% |
| Market Capitalisation | SEK 815 bn |
| Share Price | SEK 166 |

# Tools

### Programming

- Python

### Libraries

- NumPy
- Pandas
- SciPy
- Matplotlib

### Methods

- CAPM
- WACC
- ROIC Analysis
- Economic Profit
- Monte Carlo Simulation
- Brent Root Finding Algorithm

# Methodology

## Phase 1 - Cost of Capital

The analysis begins with a full CAPM-derived WACC calculation.

The model estimates:

- Cost of Equity
- Cost of Debt
- Capital Structure
- Tax Shield
- Market Value Weights

### Results

| Metric | Value |
|---------|-------:|
| Cost of Equity | 8.00% |
| After-tax Cost of Debt | 3.12% |
| WACC | **7.91%** |

Atlas Copco is almost entirely equity financed, with debt representing only 1.8% of enterprise value.

## Phase 2 - ROIC & Economic Profit

Instead of assuming arbitrary growth rates, the project derives a sustainable growth rate directly from the firm's economics.

The model calculates:

- NOPAT
- Invested Capital
- ROIC
- Economic Profit
- Reinvestment Rate

### Results

| Metric | Value |
|---------|-------:|
| ROIC | 26.7% |
| WACC | 7.9% |
| Economic Profit | SEK 19.2 bn |
| Sustainable Growth | **4.81%** |

This ROIC-derived growth assumption is consistently applied throughout both the DCF and LBO models.

## Phase 3 - Reverse DCF

Instead of estimating value, Reverse DCF asks:

> **What growth assumptions must investors believe in order to justify today's market price?**

Enterprise value is fixed at the observed market value, while Brent's root-finding algorithm solves for the implied five-year Free Cash Flow growth rate.

### Results

| Metric | Value |
|---------|-------:|
| Market-implied Growth | **18.68%** |
| Fundamental Growth | **4.81%** |

Gap:

**+13.9 percentage points**

## Phase 4 - Fundamental DCF

Using the ROIC-derived sustainable growth rate:

| Metric | Value |
|---------|-------:|
| Enterprise Value | SEK 467.9 bn |
| Value per Share | **SEK 92.20** |

Compared with the current market price of **SEK 166**, the model suggests the shares trade approximately **44.5% above** the base-case intrinsic value.

## Phase 5 - Monte Carlo Simulation

To account for valuation uncertainty, the Reverse DCF is repeated across **5,000 Monte Carlo simulations** using different combinations of WACC and terminal growth assumptions.

### Results

| Statistic | Value |
|------------|-------:|
| Mean Implied Growth | 18.05% |
| Median | 18.45% |
| 5th–95th Percentile | 8.06% - 26.89% |

## Phase 6 - Sum-of-the-Parts (SOTP)

Each operating segment is valued independently using peer EV/EBIT multiples.

Segments:

- Compressor Technique
- Vacuum Technique
- Industrial Technique
- Power Technique

### Results

| Metric | Value |
|---------|-------:|
| Value per Share | **SEK 125.19** |

Approximately **25% below** the current market valuation.

## Phase 7 - Leveraged Buyout Analysis

The project estimates the maximum acquisition price that allows a financial sponsor to achieve an **18% target IRR**.

The operating model includes:

- EBITDA-to-Free Cash Flow bridge
- Interest expense
- Taxes
- Capital expenditure
- Working capital
- Mandatory debt amortisation
- Excess cash flow sweep

### Results

| Metric | Value |
|---------|-------:|
| Maximum Entry Multiple | 7.2× EBITDA |
| Maximum Offer Price | **SEK 55.72/share** |

The analysis demonstrates why Atlas Copco is not a natural leveraged buyout candidate despite its outstanding operating performance.


## Phase 8 – IRR Bridge

Private equity returns are decomposed into their underlying drivers:

- EBITDA Growth
- Multiple Expansion
- Deleveraging

The project analyses both a breakeven transaction and an illustrative discounted acquisition to show exactly where value creation originates.


## Phase 9 - Earnings Power Value (EPV)

Following Bruce Greenwald's methodology, the project estimates:

- Reproduction Value
- Earnings Power Value
- Franchise Value

Unlike book value, invested capital includes capitalised:

- Research & Development
- Brand-building SG&A

### Results

| Metric | Value |
|---------|-------:|
| EPV | SEK 67.02/share |
| Reproduction Value | SEK 17.71/share |
| Franchise Value | SEK 49.31/share |

The results indicate that a significant portion of Atlas Copco's valuation reflects durable competitive advantages rather than tangible operating assets.


# Key Findings

- The current market price implies approximately **18.7% annual Free Cash Flow growth**, compared with a fundamentally-derived sustainable growth rate of **4.8%**.
- Reverse DCF highlights a substantial disconnect between market expectations and the company's underlying operating economics.
- A fundamentals-driven DCF produces an intrinsic value approximately **45% below** the current share price.
- Sum-of-the-Parts valuation also indicates a meaningful premium embedded in the market valuation.
- Atlas Copco's exceptional quality and low leverage make it an unattractive traditional leveraged buyout target despite its strong cash generation.
- Economic Profit analysis demonstrates significant value creation, with ROIC exceeding WACC by almost **19 percentage points**.
- EPV analysis shows that a large portion of shareholder value comes from the company's competitive moat rather than replacement assets.

# Charts Generated

The project automatically produces publication-quality visualisations including:

- Valuation Summary Dashboard
- Reverse DCF Monte Carlo Distribution
- Sum-of-the-Parts by Business Segment
- ROIC vs WACC Comparison
- LBO IRR Bridge
- EPV vs Reproduction Value


# Concepts Demonstrated

- Equity Valuation
- Corporate Finance
- Investment Banking
- Private Equity
- Discounted Cash Flow (DCF)
- Reverse DCF
- Sum-of-the-Parts (SOTP)
- Leveraged Buyout (LBO)
- Earnings Power Value (EPV)
- CAPM
- WACC
- ROIC
- Economic Profit
- Monte Carlo Simulation
- Enterprise Value
- Root Finding
- Sensitivity Analysis
- Financial Modelling in Python

---

# How to Run

Install dependencies:

```bash
pip install numpy pandas scipy matplotlib
```

Run:

```bash
python atlas_valuation.py
```

The script automatically generates all valuation charts and exports them to the project directory.


# Future Improvements

Potential extensions include:

- Comparable Company Analysis (Trading Comps)
- Precedent Transactions Analysis
- Dynamic Debt Schedule in the LBO
- Multi-stage DCF
- Scenario-based operating forecasts
- Bloomberg or Refinitiv API integration
- Interactive valuation dashboard with Streamlit

## Author

**Oresti Janko**

BSc Statistics and Insurance Science

**Interests:** Private Equity, Investment Banking, Valuation, Quantitative Finance, Financial Modelling
