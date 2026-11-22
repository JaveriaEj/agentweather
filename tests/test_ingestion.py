import pytest
import requests

from src.ingestion.aqicn import parse_feed
from src.ingestion.openweather import parse_air_item, parse_onecall_point
from src.ingestion.retry import ApiError, get_json
from src.monitoring.logs import RunLog


class FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._p, self.headers, self.text = status, payload, headers or {}, str(payload)

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_retry_on_rate_limit_and_connection_error_then_success():
    s = FakeSession([FakeResp(429, {}, {"Retry-After": "0"}), requests.ConnectionError("down"),
                     FakeResp(200, {"ok": 1})])
    assert get_json("http://x", {"appid": "secret"}, "test", session=s, sleep=lambda _: None) == {"ok": 1}
    assert s.calls == 3


def test_no_retry_on_bad_key_and_failure_is_logged_without_secret():
    log = RunLog("t")
    s = FakeSession([FakeResp(401, {"message": "Invalid API key"})])
    with pytest.raises(ApiError) as e:
        get_json("http://x", {"appid": "secret"}, "openweather", run_log=log, session=s, sleep=lambda _: None)
    assert e.value.status == 401 and s.calls == 1
    row = log.rows["api_failures"][0]
    assert "secret" not in row["params"] and row["source"] == "openweather"


def test_parse_onecall_and_air():
    p = {"dt": 1700000000, "temp": 25.1, "humidity": 60, "pressure": 1010, "dew_point": 16.0, "clouds": 20,
         "uvi": 5.2, "wind_speed": 3.4, "wind_deg": 200, "rain": {"1h": 1.5}, "weather": [{"main": "Rain"}]}
    r = parse_onecall_point(p)
    assert r["ts"] % 3600 == 0 and r["precip_mm"] == 1.5 and r["uvi"] == 5.2
    a = parse_air_item({"dt": 1700000000, "main": {"aqi": 3}, "components": {"pm2_5": 40.0, "pm10": 80.0}})
    assert a["pm2_5"] == 40.0 and a["ow_aqi_index"] == 3


def test_parse_aqicn_feed_handles_missing_values():
    payload = {"status": "ok", "data": {"aqi": "-", "iaqi": {"pm25": {"v": 120}},
                                        "time": {"iso": "2026-11-14T10:00:00+05:00"}, "city": {"name": "X"}}}
    r = parse_feed(payload)
    assert r["aqi_aqicn"] is None and r["aqicn_pm25_index"] == 120.0 and r["ts"] % 3600 == 0
    with pytest.raises(ApiError):
        parse_feed({"status": "error", "data": "Invalid key"})


def test_parse_open_meteo_hourly():
    from src.ingestion.openmeteo import parse_hourly

    payload = {"hourly": {"time": [1700000000 - 3600, 1700000000, 1700003600],
                          "temperature_2m": [20.0, 21.5, None], "relative_humidity_2m": [60, 62, None],
                          "dew_point_2m": [12, 13, None], "pressure_msl": [1010, 1011, None],
                          "cloud_cover": [10, 20, None], "precipitation": [0.0, 1.2, None],
                          "uv_index": [0.0, 4.5, None], "wind_speed_10m": [3.0, 4.0, None],
                          "wind_direction_10m": [200, 210, None]}}
    rows = parse_hourly(payload, 1700000000 - 3600, 1700003600)
    assert len(rows) == 2                                  # the hour without data is skipped
    assert rows[1]["temp_c"] == 21.5 and rows[1]["uvi"] == 4.5 and rows[1]["precip_mm"] == 1.2
    assert all(r["ts"] % 3600 == 0 for r in rows)
