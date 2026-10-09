from traffic_rl.safety import SafetyGateway


def test_gateway_enforces_minimum_green_and_protected_modes() -> None:
    gateway = SafetyGateway(minimum_green=20, heartbeat_timeout=2)
    gateway.register_signal("A", 0, 0)
    gateway.heartbeat(19)
    assert not gateway.request("A", 1, 19).accepted
    gateway.heartbeat(21)
    assert not gateway.request("A", 1, 21, "pedestrian").accepted
    assert gateway.request("A", 1, 21).accepted


def test_gateway_falls_back_on_stale_or_invalid_input() -> None:
    gateway = SafetyGateway()
    gateway.register_signal("A", 0, 0)
    gateway.heartbeat(0)
    assert not gateway.request("A", 9, 1).accepted
    assert gateway.watchdog(3)
    result = gateway.request("A", 1, 3)
    assert not result.accepted
    assert result.fallback_to_native
