from backend.app.models import Coordinate
from backend.app.processor import haversine, interpolate


def test_route_math_and_reverse_safe_interpolation():
    route = [
        Coordinate(lat=48.73023425632304, lng=9.281018887819329),
        Coordinate(lat=48.73034436371159, lng=9.279670968974454),
    ]
    length = haversine(route[0], route[1])
    midpoint = interpolate(route, 0.5)
    assert 99 < length < 101
    assert route[0].lat < midpoint["lat"] < route[1].lat
    assert route[1].lng < midpoint["lng"] < route[0].lng
