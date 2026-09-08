from __future__ import annotations

import pytest

from alpha_desk.strategies.microstructure.market_maker import (
    optimal_quotes,
    optimal_spread,
    reservation_price,
)


class TestReservationPrice:
    def test_zero_inventory_gives_reservation_price_equal_to_mid(self):
        r = reservation_price(mid_price=100.0, inventory=0.0, gamma=0.1, sigma=0.02, time_remaining=0.5)
        assert r == pytest.approx(100.0)

    def test_long_inventory_pulls_reservation_price_below_mid(self):
        r = reservation_price(mid_price=100.0, inventory=50.0, gamma=0.1, sigma=0.02, time_remaining=0.5)
        assert r < 100.0

    def test_short_inventory_pushes_reservation_price_above_mid(self):
        r = reservation_price(mid_price=100.0, inventory=-50.0, gamma=0.1, sigma=0.02, time_remaining=0.5)
        assert r > 100.0

    def test_zero_time_remaining_gives_reservation_price_equal_to_mid_regardless_of_inventory(self):
        # no more session left -> no more inventory risk to manage -> quote
        # exactly at mid, whatever the current position.
        r = reservation_price(mid_price=100.0, inventory=1000.0, gamma=0.5, sigma=0.05, time_remaining=0.0)
        assert r == pytest.approx(100.0)

    def test_higher_risk_aversion_increases_the_skew(self):
        low = reservation_price(mid_price=100.0, inventory=50.0, gamma=0.05, sigma=0.02, time_remaining=0.5)
        high = reservation_price(mid_price=100.0, inventory=50.0, gamma=0.5, sigma=0.02, time_remaining=0.5)
        assert (100.0 - high) > (100.0 - low)

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError):
            reservation_price(100.0, 0.0, gamma=0.0, sigma=0.02, time_remaining=0.5)
        with pytest.raises(ValueError):
            reservation_price(100.0, 0.0, gamma=0.1, sigma=0.02, time_remaining=-1)


class TestOptimalSpread:
    def test_higher_risk_aversion_widens_spread(self):
        # NOTE on parameter choice: the AS spread has two competing terms --
        # gamma*sigma^2*T (inventory risk, increasing in gamma) and
        # (2/gamma)*ln(1+gamma/kappa) (a liquidity-provision term that
        # DECREASES in gamma over a wide middle range). A numeric scan at
        # sigma=0.3, T=1.0, kappa=1.5 shows the spread actually DIPS to a
        # minimum around gamma~2-5 before increasing again -- confirmed
        # directly, not assumed. gamma=10 vs gamma=100 are both well past
        # that minimum, safely in the inventory-risk-dominated,
        # monotonically-increasing regime the textbook "more risk-averse ->
        # wider spread" result actually describes.
        low = optimal_spread(gamma=10.0, sigma=0.3, time_remaining=1.0, kappa=1.5)
        high = optimal_spread(gamma=100.0, sigma=0.3, time_remaining=1.0, kappa=1.5)
        assert high > low

    def test_higher_kappa_narrows_spread(self):
        # higher kappa = arrival intensity falls off faster with distance =
        # a less patient/thinner market -> the maker must quote tighter to
        # get filled at all.
        low_kappa = optimal_spread(gamma=0.1, sigma=0.02, time_remaining=0.5, kappa=0.5)
        high_kappa = optimal_spread(gamma=0.1, sigma=0.02, time_remaining=0.5, kappa=5.0)
        assert high_kappa < low_kappa

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError):
            optimal_spread(gamma=-1, sigma=0.02, time_remaining=0.5, kappa=1.0)
        with pytest.raises(ValueError):
            optimal_spread(gamma=0.1, sigma=0.02, time_remaining=0.5, kappa=0.0)


class TestOptimalQuotes:
    def test_bid_below_ask_and_centered_on_reservation_price(self):
        bid, ask = optimal_quotes(mid_price=100.0, inventory=20.0, gamma=0.1, sigma=0.02, time_remaining=0.5, kappa=1.5)
        r = reservation_price(100.0, 20.0, 0.1, 0.02, 0.5)
        spread = optimal_spread(0.1, 0.02, 0.5, 1.5)
        assert bid < ask
        assert bid == pytest.approx(r - spread / 2)
        assert ask == pytest.approx(r + spread / 2)

    def test_zero_inventory_gives_symmetric_quotes_around_mid(self):
        bid, ask = optimal_quotes(mid_price=100.0, inventory=0.0, gamma=0.1, sigma=0.02, time_remaining=0.5, kappa=1.5)
        assert (100.0 - bid) == pytest.approx(ask - 100.0)
