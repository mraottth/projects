import pytest
from fastapi.testclient import TestClient

from jobhunt.api.main import FRONTEND_DIST, app

requires_built_frontend = pytest.mark.skipif(
    not FRONTEND_DIST.exists(), reason="frontend/dist not built (run `make frontend`)"
)


@requires_built_frontend
def test_direct_navigation_to_a_client_route_serves_index_html():
    # Regression: StaticFiles(html=True) alone only serves index.html for "/" and real
    # directories, not arbitrary react-router paths -- every page but Digest 404'd on a direct
    # navigation or refresh before the SPA fallback route was added.
    with TestClient(app) as client:
        for path in ["/profile", "/tracker", "/sources", "/settings", "/jobs/123"]:
            resp = client.get(path)
            assert resp.status_code == 200, path
            assert "<div id=\"root\">" in resp.text


@requires_built_frontend
def test_unknown_api_path_still_404s_instead_of_falling_back_to_index_html():
    with TestClient(app) as client:
        resp = client.get("/api/this-does-not-exist")
        assert resp.status_code == 404


@requires_built_frontend
def test_real_asset_path_is_served_directly():
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
