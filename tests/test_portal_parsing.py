from datetime import datetime

from eplanes.acquisition.portal import parse_date, parse_flight_ids, parse_flight_page

AIRCRAFT_HTML = """
<html><body><table>
<tr class="clickable-aircraft" onclick="details('../flight/', 7301)"><td>a</td></tr>
<tr class="clickable-aircraft" onclick="details('flight/', 7302)"><td>b</td></tr>
</table><span class="current">Page 1 of 3</span></body></html>"""

FLIGHT_HTML = """
<html><body><table class="white-link">
<tr><td>Date</td><td>Aug. 28, 2024, 5:52 p.m.</td></tr>
<tr><td>Flight type</td><td>Flight test</td></tr>
<tr><td>Serial number</td><td>VSW1280102</td></tr></table>
<a href="/electro/csv/29180-7315.zip">Download CSV file</a></body></html>"""


def test_parse_flight_ids_and_pages():
    ids, total = parse_flight_ids(AIRCRAFT_HTML)
    assert ids == [7301, 7302] and total == 3


def test_parse_flight_page():
    p = parse_flight_page(7315, FLIGHT_HTML, "https://cloud.pipistrel.si")
    assert p.download_url == "https://cloud.pipistrel.si/electro/csv/29180-7315.zip"
    assert p.info["Flight type"] == "Flight test" and p.info["Serial number"] == "VSW1280102"


def test_parse_date_variants():
    assert parse_date("Aug. 28, 2024, 5:52 p.m.") == datetime(2024, 8, 28, 17, 52)
    assert parse_date("July 22, 2026, midnight") == datetime(2026, 7, 22, 0, 0)
    assert parse_date("March 3, 2025, noon") == datetime(2025, 3, 3, 12, 0)
    assert parse_date("June 3, 2023, 9 a.m.") == datetime(2023, 6, 3, 9, 0)
    assert parse_date("garbage") is None


def test_login_page_detection():
    from eplanes.acquisition.portal import looks_like_login_page
    assert looks_like_login_page('<form><input name="username"><input type="password" name="password"></form>')
    assert not looks_like_login_page(AIRCRAFT_HTML)
