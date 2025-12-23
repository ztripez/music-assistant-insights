# Music Assistant Insight Sidecar

A lightweight, high-performance inference sidecar for Music Assistant that provides audio and text embeddings using CLAP (Contrastive Language-Audio Pretraining) models.

## Overview

This sidecar offloads ML inference from Music Assistant, keeping the main server free of heavy ML/CUDA dependencies. It provides:

- **Text embeddings** from track metadata (artist, album, genre, mood)
- **Audio embeddings** from PCM audio frames
- **Vector storage** for similarity search and recommendations
- **User interaction tracking** for personalized recommendations

## Architecture

```
┌─────────────────────┐         ┌─────────────────────────────┐
│   Music Assistant   │  HTTP   │      Insight Sidecar        │
│                     │◄───────►│                             │
│  - Library sync     │ msgpack │  - CLAP inference (ONNX)    │
│  - Playback events  │   +     │  - Vector storage (Qdrant)  │
│  - Search/similar   │  WS     │  - Recommendations          │
└─────────────────────┘         └─────────────────────────────┘
```

## Features

- **Semantic search**: Find tracks by natural language queries ("upbeat electronic for working out")
- **Similar tracks**: Find tracks similar to a given track based on audio/text embeddings
- **Recommendations**: Personalized recommendations based on listening history
- **Real-time processing**: Stream audio during playback for embedding extraction

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Rust | Single binary, no GIL, minimal resources |
| ML Runtime | ort (ONNX Runtime) | CUDA support, 3-5x faster than Python |
| Vector DB | Qdrant (embedded) | Rust-native, fast filtered queries |
| HTTP Server | axum | Async, WebSocket support |
| Serialization | msgpack (rmp-serde) | Efficient binary transport |
| Audio | symphonia | Pure Rust audio decoding |

## Models

Uses pre-exported ONNX CLAP models from Hugging Face:

| Model | Size | Use Case |
|-------|------|----------|
| `Xenova/clap-htsat-unfused` | ~150MB | Default, good balance |
| `laion/larger_clap_music` | ~200MB | Music-tuned, better quality |

Embedding dimension: 512 (float32)

## API

### Transport

- HTTP with msgpack bodies
- WebSocket for real-time audio streaming
- Content-Type: `application/msgpack`

### Endpoints

#### Tracks

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/tracks/upsert` | Upsert track with metadata |
| `POST` | `/api/v1/tracks/upsert/batch` | Batch upsert for library sync |
| `DELETE` | `/api/v1/tracks/{track_id}` | Remove track |
| `POST` | `/api/v1/tracks/{track_id}/audio` | Add audio embedding |

#### Search & Similarity

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/search` | Semantic text search |
| `POST` | `/api/v1/similar` | Find similar tracks |

#### Interactions & Recommendations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/interactions` | Record play/skip/favorite |
| `POST` | `/api/v1/recommendations` | Get personalized recommendations |

#### Streaming

| Method | Path | Description |
|--------|------|-------------|
| `WS` | `/api/v1/stream/audio` | Real-time audio embedding |

#### Admin

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/config` | Current configuration |
| `POST` | `/api/v1/admin/reload` | Hot-reload model |
| `DELETE` | `/api/v1/admin/reset` | Wipe all data |

### Data Types

```rust
// Track metadata for upsert
struct TrackMetadata {
    name: String,
    artists: Vec<String>,
    album: Option<String>,
    genres: Vec<String>,
    mood: Option<String>,
    isrc: Option<String>,
    duration_ms: Option<u64>,
}

// Audio data
struct AudioData {
    format: AudioFormat,      // Pcm_f32le, Pcm_s16le, Pcm_s24le
    sample_rate: u32,         // 44100, 48000, etc.
    channels: u8,             // 1 (mono) or 2 (stereo)
    data: Vec<u8>,            // Raw PCM bytes
}

// Search/similarity blend modes
enum BlendMode {
    Text,      // Text embeddings only
    Audio,     // Audio embeddings only
    Combined,  // Both (default)
}

// User interactions
enum InteractionType {
    Played,
    Skipped,
    Favorited,
    Unfavorited,
}
```

## Configuration

Environment variables:

```bash
# Model
INSIGHT_MODEL=Xenova/clap-htsat-unfused
INSIGHT_ENABLE_CUDA=true

# Audio processing
INSIGHT_WINDOW_SIZE_S=10.0
INSIGHT_HOP_SIZE_S=10.0

# Storage
INSIGHT_STORAGE_PATH=/data/qdrant

# Server
INSIGHT_HOST=0.0.0.0
INSIGHT_PORT=8096
```

## Deployment

### Docker

```yaml
services:
  insight-sidecar:
    image: ghcr.io/music-assistant/insight-sidecar:latest
    environment:
      - INSIGHT_MODEL=Xenova/clap-htsat-unfused
      - INSIGHT_ENABLE_CUDA=true
      - INSIGHT_STORAGE_PATH=/data/qdrant
    volumes:
      - insight-data:/data
    ports:
      - "8096:8096"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  insight-data:
```

### Docker Compose with Music Assistant

```yaml
services:
  music-assistant:
    image: ghcr.io/music-assistant/server:latest
    environment:
      - INSIGHT_SIDECAR_URL=http://insight-sidecar:8096
    depends_on:
      insight-sidecar:
        condition: service_healthy

  insight-sidecar:
    image: ghcr.io/music-assistant/insight-sidecar:latest
    volumes:
      - insight-data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8096/api/v1/health"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  insight-data:
```

### Standalone Binary

```bash
# Download
curl -LO https://github.com/music-assistant/insight-sidecar/releases/latest/download/insight-sidecar-linux-x86_64

# Run
INSIGHT_STORAGE_PATH=./data ./insight-sidecar-linux-x86_64
```

## Resource Requirements

| Configuration | CPU | Memory | Storage |
|---------------|-----|--------|---------|
| Minimal (CPU only) | 2 cores | 512MB | 500MB + 1KB/track |
| Recommended | 4 cores | 1GB | 500MB + 1KB/track |
| With CUDA | 4 cores + GPU | 2GB | 500MB + 1KB/track |

Inference latency:
- Text embedding: ~10-30ms (CPU), ~5-10ms (CUDA)
- Audio embedding (10s window): ~50-100ms (CPU), ~20-40ms (CUDA)

## Development

### Prerequisites

- Rust 1.75+
- ONNX Runtime (automatically downloaded)
- (Optional) CUDA toolkit for GPU support

### Building

```bash
# CPU only
cargo build --release

# With CUDA
cargo build --release --features cuda
```

### Testing

```bash
cargo test

# Integration tests (requires running instance)
cargo test --features integration
```

### Project Structure

```
insight-sidecar/
├── src/
│   ├── main.rs           # Entry point
│   ├── config.rs         # Configuration
│   ├── server/
│   │   ├── mod.rs
│   │   ├── routes.rs     # HTTP routes
│   │   └── websocket.rs  # WebSocket handler
│   ├── inference/
│   │   ├── mod.rs
│   │   ├── clap.rs       # CLAP model wrapper
│   │   └── audio.rs      # Audio preprocessing
│   ├── storage/
│   │   ├── mod.rs
│   │   └── qdrant.rs     # Vector storage
│   └── recommendations/
│       ├── mod.rs
│       └── engine.rs     # Recommendation logic
├── models/               # ONNX models (downloaded at build/runtime)
├── Cargo.toml
├── Dockerfile
└── README.md
```

## License

Apache-2.0

## Related Projects

- [Music Assistant](https://github.com/music-assistant/server) - The main Music Assistant server
- [LAION CLAP](https://github.com/LAION-AI/CLAP) - The CLAP model architecture
- [Qdrant](https://github.com/qdrant/qdrant) - Vector database
