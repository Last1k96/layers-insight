"""API integration tests for graph endpoints."""
from __future__ import annotations

import json

import pytest


class TestGetGraph:
    @pytest.mark.asyncio
    async def test_graph_with_cache(self, async_client, test_session, test_app):
        svc = test_app.state.session_service
        graph_data = {
            "nodes": [{"id": "n1", "name": "n1", "type": "Relu", "shape": [1, 3]}],
            "edges": [],
        }
        svc.save_graph_cache(test_session.id, graph_data)

        resp = await async_client.get(f"/api/sessions/{test_session.id}/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"][0]["id"] == "n1"

    @pytest.mark.asyncio
    async def test_graph_no_ov_no_cache(self, async_client, test_session):
        """No OV and no cache should return 503."""
        resp = await async_client.get(f"/api/sessions/{test_session.id}/graph")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_graph_session_not_found(self, async_client):
        resp = await async_client.get("/api/sessions/nonexistent/graph")
        assert resp.status_code == 404


class TestSearchGraph:
    @pytest.mark.asyncio
    async def test_search_matching(self, async_client, test_session, test_app):
        svc = test_app.state.session_service
        graph_data = {
            "nodes": [
                {"id": "conv_0", "name": "conv1", "type": "Convolution",
                 "shape": [1, 64], "category": "Convolution", "color": "#335588",
                 "attributes": {}, "inputs": [], "width": 100, "height": 32},
                {"id": "relu_0", "name": "relu1", "type": "Relu",
                 "shape": [1, 64], "category": "Activation", "color": "#702921",
                 "attributes": {}, "inputs": [], "width": 100, "height": 32},
            ],
            "edges": [],
        }
        svc.save_graph_cache(test_session.id, graph_data)

        resp = await async_client.get(
            f"/api/sessions/{test_session.id}/graph/search",
            params={"q": "conv"},
        )
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["name"] == "conv1"

    @pytest.mark.asyncio
    async def test_search_empty_query(self, async_client, test_session, test_app):
        svc = test_app.state.session_service
        graph_data = {
            "nodes": [
                {"id": "conv_0", "name": "conv1", "type": "Convolution",
                 "shape": [1, 64], "category": "Convolution", "color": "#335588",
                 "attributes": {}, "inputs": [], "width": 100, "height": 32},
            ],
            "edges": [],
        }
        svc.save_graph_cache(test_session.id, graph_data)

        resp = await async_client.get(
            f"/api/sessions/{test_session.id}/graph/search",
            params={"q": ""},
        )
        assert resp.status_code == 200
        # Empty query matches all
        assert len(resp.json()) == 1

    @pytest.mark.asyncio
    async def test_search_limit(self, async_client, test_session, test_app):
        """Search results are limited to 50."""
        svc = test_app.state.session_service
        nodes = [
            {"id": f"n_{i}", "name": f"node_{i}", "type": "Relu",
             "shape": [1], "category": "Activation", "color": "#702921",
             "attributes": {}, "inputs": [], "width": 100, "height": 32}
            for i in range(60)
        ]
        svc.save_graph_cache(test_session.id, {"nodes": nodes, "edges": []})

        resp = await async_client.get(
            f"/api/sessions/{test_session.id}/graph/search",
            params={"q": "node"},
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 50


class TestGetConstantData:
    @pytest.mark.asyncio
    async def test_constant_model_not_loaded(self, async_client, test_session):
        resp = await async_client.get(
            f"/api/sessions/{test_session.id}/graph/constant/some_const"
        )
        assert resp.status_code == 404
