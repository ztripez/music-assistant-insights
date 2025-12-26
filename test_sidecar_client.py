#!/usr/bin/env python3
"""
Minimal SidecarClient for Testing
Purpose: Standalone client extracted from MA provider for integration tests
Date: 2025-12-26

This is a minimal copy of the SidecarClient from the Music Assistant provider,
containing only the methods needed for Phase 1 integration testing. This avoids
requiring the full Music Assistant package and its dependencies.
"""

from dataclasses import dataclass
from typing import Any

import aiohttp
import msgpack


@dataclass
class TrackMetadata:
    """Metadata for a track to be embedded."""

    name: str
    artists: list[str]
    album: str | None = None
    genres: list[str] | None = None


class SidecarClient:
    """
    Async HTTP client for the Music Assistant Insight Sidecar.

    Communicates with the Rust sidecar using MessagePack serialization
    for efficient binary transfer of embeddings.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8096",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the sidecar client.

        :param base_url: Base URL of the sidecar API.
        :param timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={"Content-Type": "application/msgpack"},
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    @staticmethod
    async def _get_error_body(resp: aiohttp.ClientResponse) -> str:
        """Safely extract error body from response (msgpack or text)."""
        try:
            data = await resp.read()
            # Try to decode as msgpack first
            try:
                result = msgpack.unpackb(data, raw=False)
                if isinstance(result, dict):
                    return str(result.get("error", result))
                return str(result)
            except Exception:
                # Fall back to text decoding
                return data.decode("utf-8", errors="replace")
        except Exception:
            return "<failed to read response body>"

    async def health_check(self) -> dict[str, Any]:
        """
        Check if the sidecar is healthy.

        :return: Health status dict with 'status', 'model_loaded', 'storage_ready'.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/health"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise ConnectionError(f"Sidecar health check failed: {resp.status}")
            data = await resp.read()
            result: dict[str, Any] = msgpack.unpackb(data, raw=False)
            return result

    async def embed_text_and_store(
        self,
        track_id: str,
        metadata: TrackMetadata,
    ) -> dict[str, Any]:
        """
        Generate text embedding from metadata and store it in one operation.

        :param track_id: Unique track identifier.
        :param metadata: Track metadata for embedding.
        :return: Response with track_id, stored status, and embedded text.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/tracks/embed-text"

        payload = {
            "track_id": track_id,
            "metadata": {
                "name": metadata.name,
                "artists": metadata.artists,
                "album": metadata.album,
                "genres": metadata.genres or [],
            },
        }

        data = msgpack.packb(payload)
        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(
                    f"Failed to embed and store track: {resp.status} - {body}"
                )
            result: dict[str, Any] = msgpack.unpackb(await resp.read(), raw=False)
            return result

    async def compute_taste_profile(
        self,
        user_id: str,
        interactions: list[dict[str, Any]],
        cutoff_days: int = 21,
    ) -> dict[str, Any]:
        """
        Compute taste profile from user interactions.

        :param user_id: User ID to compute profile for.
        :param interactions: List of interaction dicts.
        :param cutoff_days: Number of days of history to consider.
        :return: Response with user_id and profile metadata.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/users/{user_id}/profile/compute"

        payload = {
            "interactions": interactions,
            "cutoff_days": cutoff_days,
            "profile_type": "global",  # This uses ProfileTypeRequest which is lowercase
        }

        data = msgpack.packb(payload)
        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to compute taste profile: {resp.status} - {body}")
            result: dict[str, Any] = msgpack.unpackb(await resp.read(), raw=False)
            return result

    async def get_taste_recommendations(
        self,
        user_id: str,
        limit: int = 25,
        exclude_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Get personalized recommendations based on taste profile.

        :param user_id: User ID to get recommendations for.
        :param limit: Maximum number of recommendations.
        :param exclude_ids: Track IDs to exclude from results.
        :return: Response with tracks list and profile_confidence.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/users/{user_id}/recommend"

        payload = {
            "limit": limit,
            "profile_type": {"type": "Global"},  # Adjacently tagged enum format
            "exclude_ids": exclude_ids or [],
            "filter": {},
        }

        data = msgpack.packb(payload)
        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to get recommendations: {resp.status} - {body}")
            result: dict[str, Any] = msgpack.unpackb(await resp.read(), raw=False)
            return result
