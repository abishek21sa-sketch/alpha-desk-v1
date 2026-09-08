# Data Dictionary

Plain-English companion to `data/contracts/*.json`. If a term here and the
contract disagree, the contract is authoritative (this file is for humans,
the contract is for validation code).

## `prices` table

Daily OHLCV for 40 individual equities + 10 ETF proxies (`alpha_desk.data.universe`).
Values are **split- and dividend-adjusted** (`auto_adjust=True`), meaning a
1980 AAPL close is on the same scale as a 2026 AAPL close despite four stock
splits in between -- this is what backtests should use; it is not what you'd
see quoted on an exchange ticker that day.

## `macro` table

Nine FRED series covering the yield curve (3M/2Y/10Y/10Y-2Y spread), policy
rate (effective fed funds), volatility (VIX), credit risk appetite (HY OAS
spread), and the broader macro backdrop (CPI, unemployment). These are the
regime/context inputs later modules (momentum, risk) condition on -- they are
not tradeable instruments themselves (VIXCLS is the *index* level, not a
tradeable VIX future).

## `fundamentals` table

Eight XBRL line items pulled per company from SEC EDGAR's company-facts API:

| Concept | What it is |
|---|---|
| `Revenues` | Total reported revenue for the period |
| `NetIncomeLoss` | Bottom-line net income (or loss) |
| `Assets` | Total assets (balance-sheet instant, not a period flow) |
| `Liabilities` | Total liabilities (balance-sheet instant) |
| `StockholdersEquity` | Book value of equity (balance-sheet instant) |
| `EarningsPerShareDiluted` | Diluted EPS as reported |
| `CommonStockSharesOutstanding` | Shares outstanding as of the reporting date |
| `OperatingIncomeLoss` | Operating income before interest/tax adjustments |

**Not every company tags every concept identically** -- XBRL tagging is
filer-discretionary, so a smaller/older filer may be missing a concept entirely,
or tag an equivalent line item under a different name this extractor doesn't
look for. `alpha_desk.data.sources.XBRL_CONCEPTS` is the exact allowlist; a
concept not on that list is silently not extracted (this is a scope choice,
not a bug -- see `docs/DATA_BACKBONE.md`).

**`filed` vs `end_date`**: `end_date` is the accounting period the number
describes (e.g. fiscal Q3 ending 2026-06-30); `filed` is the date that number
actually became public knowledge via EDGAR (e.g. 2026-08-04). Any model using
this data to predict or trade *must* condition on `filed`, not `end_date` --
using `end_date` would let a signal "see" a quarter's financials before the
company had actually reported them, which is a textbook look-ahead-bias bug.

## `universe` table

Static metadata (sector / asset class) for every symbol in `prices`, joined
in for factor-model grouping. Not a dataset in its own right -- generated
from `alpha_desk.data.universe`, not fetched from an external source.
