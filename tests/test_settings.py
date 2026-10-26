from jobhunt.core.settings import get_setting, get_settings, update_settings


def test_defaults_are_seeded():
    assert get_setting("daily_llm_cap") == 50


def test_update_settings_upserts_and_persists():
    update_settings({"daily_llm_cap": 10, "prefilter_locations": ["Remote"]})
    assert get_setting("daily_llm_cap") == 10
    assert get_setting("prefilter_locations") == ["Remote"]

    update_settings({"daily_llm_cap": 25})
    assert get_setting("daily_llm_cap") == 25
    assert get_setting("prefilter_locations") == ["Remote"]  # untouched keys survive


def test_get_setting_default_for_missing_key():
    assert get_setting("nonexistent_key", "fallback") == "fallback"


def test_get_settings_returns_everything():
    settings = get_settings()
    assert "daily_llm_cap" in settings
