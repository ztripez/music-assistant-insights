# Music Assistant Integration Flow

This document describes how Music Assistant (MA) integrates with the Insight Sidecar for audio embedding and similarity search.

## Overview

The sidecar provides two main capabilities:

1. **Ingestion** - Store track embeddings from audio/metadata
2. **Search** - Find similar tracks by text query or track ID

These are separate flows. You ingest tracks to build the database, then search against it.

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│   Music Assistant   │         │   Insight Sidecar    │
│                     │         │                      │
│  ┌───────────────┐  │  HTTP   │  ┌────────────────┐  │
│  │ Audio Player  │──┼────────►│  │ Stream API     │  │
│  │ (frames)      │  │         │  │ (buffer+embed) │  │
│  └───────────────┘  │         │  └───────┬────────┘  │
│                     │         │          │           │
│  ┌───────────────┐  │         │  ┌───────▼────────┐  │
│  │ Metadata      │──┼────────►│  │ CLAP Model     │  │
│  │ Provider      │  │         │  │ (inference)    │  │
│  └───────────────┘  │         │  └───────┬────────┘  │
│                     │         │          │           │
│  ┌───────────────┐  │         │  ┌───────▼────────┐  │
│  │ Playlist/     │◄─┼────────┤│  │ Vector DB      │  │
│  │ Recommender   │  │         │  │ (usearch)      │  │
│  └───────────────┘  │         │  └────────────────┘  │
└─────────────────────┘         └──────────────────────┘
```

## Flow 1: Streaming Ingestion

Use this when a user plays a track. Stream audio frames as they arrive.

### Step 1: Start Session

When playback begins, open an ingestion session:

```http
POST /api/v1/stream/start
Content-Type: application/msgpack

{
  "track_id": "abc123",
  "metadata": {
    "name": "Bohemian Rhapsody",
    "artists": ["Queen"],
    "album": "A Night at the Opera",
    "genres": ["Rock", "Progressive Rock"]
  },
  "format": "pcm_s16_le",
  "sample_rate": 44100,
  "channels": 2
}
```

Response (msgpack):
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "window_samples": 480000
}
```

### Step 2: Stream Frames

As audio frames arrive from the player, send them to the sidecar:

```http
POST /api/v1/stream/{session_id}/frames
Content-Type: application/octet-stream

<raw PCM bytes>
```

Response (msgpack):
```json
{
  "buffered_seconds": 4.2,
  "windows_completed": 0
}
```

**Important:**
- Send frames as they arrive (don't wait to accumulate)
- Sidecar buffers internally until 10 seconds are available
- When `windows_completed` increases, an embedding was generated

### Step 3: End Session

When playback ends (or user skips), finalize the session:

```http
POST /api/v1/stream/{session_id}/end
Content-Type: application/msgpack

{
  "store": true,
  "min_duration_s": 3.0
}
```

Response (msgpack):
```json
{
  "track_id": "abc123",
  "duration_s": 45.2,
  "windows_processed": 4,
  "text_stored": true,
  "audio_stored": true
}
```

### Abort Session

If playback is cancelled or errors occur:

```http
DELETE /api/v1/stream/{session_id}
```

### Check Session Status (Optional)

Check the status of an active session:

```http
GET /api/v1/stream/{session_id}/status
```

Response (msgpack):
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "track_id": "abc123",
  "status": "active",
  "buffered_seconds": 15.3,
  "windows_completed": 1,
  "age_seconds": 18.2
}
```

Status values: `active`, `finalizing`, `completed`, `aborted`, `error`

## Flow 2: Search

Query the database for similar tracks. Search requires an embedding vector.

### Step 1: Generate Query Embedding

For text-based search, first generate an embedding:

```http
POST /api/v1/embed/text
Content-Type: application/msgpack

{
  "text": "upbeat electronic dance music"
}
```

Response (msgpack):
```json
{
  "embedding": [0.123, -0.456, ...],  // 512-dimensional
  "text": "upbeat electronic dance music"
}
```

You can also provide metadata instead of raw text:
```json
{
  "metadata": {
    "name": "Track Name",
    "artists": ["Artist"],
    "album": "Album Name",
    "genres": ["Genre"]
  }
}
```

### Step 2: Search with Embedding

```http
POST /api/v1/tracks/search
Content-Type: application/msgpack

{
  "embedding": [0.123, -0.456, ...],
  "collection": "text",
  "limit": 10
}
```

Response (msgpack):
```json
{
  "results": [
    {
      "track_id": "def456",
      "score": 0.92,
      "metadata": {
        "name": "Don't Stop Me Now",
        "artists": ["Queen"],
        "album": "Jazz"
      }
    }
  ],
  "count": 1
}
```

### Search by Track ID (Similar Tracks)

To find similar tracks, first get the track's embedding:

```http
GET /api/v1/tracks/{track_id}?include_audio=true
```

Response (msgpack):
```json
{
  "track_id": "abc123",
  "metadata": {...},
  "has_text": true,
  "has_audio": true,
  "audio_embedding": [0.789, -0.012, ...]
}
```

Then search using that embedding in the `audio` collection:

```http
POST /api/v1/tracks/search
Content-Type: application/msgpack

{
  "embedding": [0.789, -0.012, ...],
  "collection": "audio",
  "limit": 10,
  "filter": {
    "exclude_ids": ["abc123"]
  }
}
```

## Audio Format Requirements

The sidecar accepts these PCM formats:

| Format | Description | Bytes/Sample |
|--------|-------------|--------------|
| `pcm_f32_le` | 32-bit float, little-endian | 4 |
| `pcm_s16_le` | 16-bit signed int, little-endian | 2 |
| `pcm_s24_le` | 24-bit signed int, little-endian | 3 |

**Sample rates:** Any (resampled internally to 48kHz)
**Channels:** 1 (mono) or 2 (stereo, converted to mono internally)

## Timing Considerations

- **Window size:** 10 seconds (480,000 samples at 48kHz)
- **Minimum duration:** 3 seconds (configurable) - shorter audio won't generate embedding
- **Session timeout:** 5 minutes of inactivity (configurable)

## Example: Python Integration

```python
import httpx
import msgpack

class InsightClient:
    def __init__(self, base_url="http://localhost:8096"):
        self.base_url = base_url
        self.session_id = None
        self.headers = {"Content-Type": "application/msgpack"}

    def _pack(self, data: dict) -> bytes:
        return msgpack.packb(data)

    def _unpack(self, resp: httpx.Response) -> dict:
        return msgpack.unpackb(resp.content)

    def start_stream(self, track_id: str, metadata: dict,
                     sample_rate: int = 44100, channels: int = 2):
        resp = httpx.post(
            f"{self.base_url}/api/v1/stream/start",
            content=self._pack({
                "track_id": track_id,
                "metadata": metadata,
                "format": "pcm_s16_le",
                "sample_rate": sample_rate,
                "channels": channels
            }),
            headers=self.headers
        )
        data = self._unpack(resp)
        self.session_id = data["session_id"]
        return self.session_id

    def send_frames(self, pcm_bytes: bytes):
        resp = httpx.post(
            f"{self.base_url}/api/v1/stream/{self.session_id}/frames",
            content=pcm_bytes,
            headers={"Content-Type": "application/octet-stream"}
        )
        return self._unpack(resp)

    def end_stream(self, store: bool = True):
        resp = httpx.post(
            f"{self.base_url}/api/v1/stream/{self.session_id}/end",
            content=self._pack({"store": store, "min_duration_s": 3.0}),
            headers=self.headers
        )
        self.session_id = None
        return self._unpack(resp)

    def search_by_text(self, query: str, limit: int = 10):
        # Step 1: Generate embedding from text query
        embed_resp = httpx.post(
            f"{self.base_url}/api/v1/embed/text",
            content=self._pack({"text": query}),
            headers=self.headers
        )
        embedding = self._unpack(embed_resp)["embedding"]

        # Step 2: Search with embedding
        search_resp = httpx.post(
            f"{self.base_url}/api/v1/tracks/search",
            content=self._pack({
                "embedding": embedding,
                "collection": "text",
                "limit": limit
            }),
            headers=self.headers
        )
        return self._unpack(search_resp)["results"]

    def search_similar(self, track_id: str, limit: int = 10):
        # Step 1: Get track's audio embedding
        track_resp = httpx.get(
            f"{self.base_url}/api/v1/tracks/{track_id}",
            params={"include_audio": "true"}
        )
        data = self._unpack(track_resp)
        audio_embedding = data.get("audio_embedding")
        if not audio_embedding:
            return []

        # Step 2: Search with embedding
        search_resp = httpx.post(
            f"{self.base_url}/api/v1/tracks/search",
            content=self._pack({
                "embedding": audio_embedding,
                "collection": "audio",
                "limit": limit,
                "filter": {"exclude_ids": [track_id]}
            }),
            headers=self.headers
        )
        return self._unpack(search_resp)["results"]

# Usage in MA provider
client = InsightClient()

# When playback starts
client.start_stream(
    track_id="spotify:track:abc123",
    metadata={
        "name": "Song Title",
        "artists": ["Artist Name"],
        "album": "Album Name",
        "genres": ["Pop"]
    }
)

# In audio callback (called repeatedly with frames)
def on_audio_frame(pcm_data: bytes):
    status = client.send_frames(pcm_data)
    print(f"Buffered: {status['buffered_seconds']:.1f}s")

# When playback ends
result = client.end_stream(store=True)
print(f"Stored {result['windows_processed']} windows")

# Later: find similar tracks
similar = client.search_similar("spotify:track:abc123")
for track in similar:
    print(f"{track['metadata']['name']} - {track['score']:.2f}")

# Or search by text description
results = client.search_by_text("chill acoustic guitar")
for track in results:
    print(f"{track['metadata']['name']} - {track['score']:.2f}")
```

## Error Handling

| Status | Meaning |
|--------|---------|
| 400 | Invalid request (bad format, missing fields) |
| 404 | Session not found (expired or invalid ID) |
| 409 | Session already exists for track_id |
| 503 | Model not loaded or storage unavailable |

## Configuration

Environment variables for the sidecar:

```bash
INSIGHT_STREAM__SESSION_TIMEOUT_S=300   # 5 min timeout
INSIGHT_STREAM__MAX_SESSIONS=100        # Max concurrent sessions
INSIGHT_STREAM__MIN_DURATION_S=3.0      # Min audio for embedding
```
