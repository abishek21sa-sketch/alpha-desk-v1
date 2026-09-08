from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from alpha_desk.execution.alpaca_client import AlpacaClient, AlpacaConfigError


class TestAlpacaClientConfiguration:
    def test_is_configured_true_when_both_keys_given(self):
        client = AlpacaClient(api_key="k", secret_key="s")
        assert client.is_configured() is True

    def test_is_configured_false_when_keys_missing(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        client = AlpacaClient()
        assert client.is_configured() is False

    def test_reads_credentials_from_environment_by_default(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "env-key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "env-secret")
        client = AlpacaClient()
        assert client.api_key == "env-key"
        assert client.secret_key == "env-secret"

    def test_defaults_to_the_paper_trading_host(self, monkeypatch):
        monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
        client = AlpacaClient(api_key="k", secret_key="s")
        assert client.base_url == "https://paper-api.alpaca.markets"

    def test_explicit_paper_host_is_accepted(self):
        client = AlpacaClient(api_key="k", secret_key="s", base_url="https://paper-api.alpaca.markets")
        assert "paper-api" in client.base_url

    def test_refuses_the_live_trading_host(self):
        with pytest.raises(AlpacaConfigError):
            AlpacaClient(api_key="k", secret_key="s", base_url="https://api.alpaca.markets")

    def test_calling_without_credentials_raises_a_clear_error(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        client = AlpacaClient()
        with pytest.raises(AlpacaConfigError):
            client.get_account()


class TestAlpacaClientRequests:
    def _client(self) -> AlpacaClient:
        return AlpacaClient(api_key="test-key", secret_key="test-secret")

    @patch("alpha_desk.execution.alpaca_client.requests.get")
    def test_get_account_hits_the_right_endpoint_with_auth_headers(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"equity": "100000.00", "status": "ACTIVE"}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = self._client().get_account()

        assert result["status"] == "ACTIVE"
        called_url, called_kwargs = mock_get.call_args[0][0], mock_get.call_args[1]
        assert called_url == "https://paper-api.alpaca.markets/v2/account"
        assert called_kwargs["headers"]["APCA-API-KEY-ID"] == "test-key"
        assert called_kwargs["headers"]["APCA-API-SECRET-KEY"] == "test-secret"

    @patch("alpha_desk.execution.alpaca_client.requests.get")
    def test_list_positions_returns_a_list(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"symbol": "AAPL", "qty": "10"}]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        positions = self._client().list_positions()
        assert positions == [{"symbol": "AAPL", "qty": "10"}]

    @patch("alpha_desk.execution.alpaca_client.requests.get")
    def test_http_error_propagates_rather_than_being_swallowed(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("401 Unauthorized")
        mock_get.return_value = mock_resp

        with pytest.raises(Exception, match="401"):
            self._client().get_account()

    @patch("alpha_desk.execution.alpaca_client.requests.post")
    def test_submit_order_posts_the_expected_payload(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": "order-1", "status": "accepted"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        result = self._client().submit_order(symbol="AAPL", qty=1, side="buy")

        assert result["status"] == "accepted"
        called_url = mock_post.call_args[0][0]
        called_payload = mock_post.call_args[1]["json"]
        assert called_url == "https://paper-api.alpaca.markets/v2/orders"
        assert called_payload == {
            "symbol": "AAPL", "qty": "1", "side": "buy", "type": "market", "time_in_force": "day",
        }

    def test_submit_order_rejects_invalid_side(self):
        with pytest.raises(ValueError):
            self._client().submit_order(symbol="AAPL", qty=1, side="hold")

    def test_submit_order_rejects_nonpositive_qty(self):
        with pytest.raises(ValueError):
            self._client().submit_order(symbol="AAPL", qty=0, side="buy")
