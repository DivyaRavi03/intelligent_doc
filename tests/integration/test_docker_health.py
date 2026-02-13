"""Integration test: verify all Docker services start and report healthy.

This test requires Docker and docker-compose to be available.
Run manually with:
    pytest tests/integration/test_docker_health.py -v

Not included in default CI unit test runs because it requires Docker.
"""

from __future__ import annotations

import subprocess
import time

import httpx
import pytest

COMPOSE_FILE = "docker-compose.prod.yml"
HEALTH_URL = "http://localhost:8000/api/v1/admin/health"
STARTUP_TIMEOUT = 120  # seconds
POLL_INTERVAL = 3  # seconds


@pytest.fixture(scope="module")
def docker_services():
    """Start docker-compose services and tear down after tests."""
    subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "up", "-d", "--build"],
        check=True,
        capture_output=True,
        text=True,
    )

    # Wait for the app to become healthy
    start = time.time()
    healthy = False
    while time.time() - start < STARTUP_TIMEOUT:
        try:
            resp = httpx.get(HEALTH_URL, timeout=5)
            if resp.status_code == 200:
                healthy = True
                break
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(POLL_INTERVAL)

    if not healthy:
        # Capture logs for debugging before failing
        logs = subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "logs", "--tail=50"],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "down", "-v"],
            capture_output=True,
            text=True,
        )
        pytest.fail(
            f"Services did not become healthy within {STARTUP_TIMEOUT}s.\n"
            f"{logs.stdout}\n{logs.stderr}"
        )

    yield

    subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "down", "-v"],
        check=True,
        capture_output=True,
        text=True,
    )


class TestDockerHealth:
    """Verify all services report healthy via the admin health endpoint."""

    def test_health_endpoint_returns_200(self, docker_services: None) -> None:
        """GET /api/v1/admin/health returns 200."""
        resp = httpx.get(HEALTH_URL, timeout=10)
        assert resp.status_code == 200

    def test_health_all_services_ok(self, docker_services: None) -> None:
        """Health response reports database and redis as ok."""
        resp = httpx.get(HEALTH_URL, timeout=10)
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["database"] == "ok"
        assert data["redis"] == "ok"
        assert "chromadb" in data

    def test_health_includes_version(self, docker_services: None) -> None:
        """Health response includes a version string."""
        resp = httpx.get(HEALTH_URL, timeout=10)
        data = resp.json()
        assert "version" in data
        assert len(data["version"]) > 0

    def test_health_includes_timestamp(self, docker_services: None) -> None:
        """Health response includes a timestamp."""
        resp = httpx.get(HEALTH_URL, timeout=10)
        data = resp.json()
        assert "timestamp" in data
