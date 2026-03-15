"""API integration tests for device/config endpoints."""
from __future__ import annotations

import pytest


class TestListDevices:
    @pytest.mark.asyncio
    async def test_devices_without_ov(self, async_client):
        """When OV is not available, should return ['CPU']."""
        resp = await async_client.get("/api/devices")
        assert resp.status_code == 200
        assert resp.json() == ["CPU"]


class TestGetDefaults:
    @pytest.mark.asyncio
    async def test_get_defaults(self, async_client):
        resp = await async_client.get("/api/defaults")
        assert resp.status_code == 200
        data = resp.json()
        assert data["main_device"] == "CPU"
        assert data["ref_device"] == "CPU"
        # model_path should be set from test fixture
        assert data["model_path"] is not None


class TestModelInputs:
    @pytest.mark.asyncio
    async def test_model_inputs_without_ov(self, async_client):
        """Without OV, should return 503."""
        resp = await async_client.get(
            "/api/model-inputs",
            params={"model_path": "/some/model.xml"},
        )
        assert resp.status_code == 503
