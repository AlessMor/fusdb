from tests.test_profile_source_system import _source_system


def test_source_profile_is_derived_provider():
    system, *_ = _source_system()
    assert system.derived_provider_by_output.get("n_e") is not None


def test_source_profile_is_not_default_provider():
    system, *_ = _source_system()
    assert system.default_provider_by_output.get("n_e") is None


def test_source_profile_is_in_completion_plan():
    system, *_ = _source_system()
    names = [rel.name for rel, _only_missing in system._completion_plan()]
    assert "Source profile n_e on B" in names


def test_source_profile_completion_provider_is_not_only_missing():
    system, *_ = _source_system()
    entries = [(rel.name, only_missing) for rel, only_missing in system._completion_plan()]
    assert ("Source profile n_e on B", False) in entries
