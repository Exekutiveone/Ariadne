from backend.app.problem_reasons import add_reason, list_reasons, use_reason


def test_custom_problem_reason_is_persistent_and_sorted_by_usage(tmp_path):
    created = add_reason(tmp_path, "Wurzelwerk im Fahrkorridor")
    use_reason(tmp_path, created["value"])
    use_reason(tmp_path, created["value"])

    reasons = list_reasons(tmp_path)
    assert reasons[0]["value"] == created["value"]
    assert reasons[0]["uses"] == 2


def test_custom_problem_reason_rejects_empty_and_duplicate_labels(tmp_path):
    add_reason(tmp_path, "Glatter Untergrund")
    try:
        add_reason(tmp_path, " glatter   untergrund ")
    except ValueError as exc:
        assert "existiert" in str(exc)
    else:
        raise AssertionError("Dubletten müssen abgelehnt werden")
