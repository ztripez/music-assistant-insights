# Music Assistant Insight Sidecar

A lightweight, high-performance inference sidecar for Music Assistant that provides audio and text embeddings using CLAP (Contrastive Language-Audio Pretraining) models.

## Overview

This sidecar offloads ML inference from Music Assistant, keeping the main server free of heavy ML/CUDA dependencies. It provides:

- **Text embeddings** from track metadata (artist, album, genre)
- **Audio embeddings** from PCM audio frames streamed during playback
- **Vector storage** for similarity search

## Architecture

```
┌─────────────────────┐         ┌─────────────────────────────┐
│   Music Assistant   │  HTTP   │      Insight Sidecar        │
│                     │◄───────►│                             │
│  - Audio Player     │ msgpack │  - Stream API (buffer+embed)│
│  - Metadata         │   +     │  - CLAP inference (ONNX)    │
│  - Search/similar   │  PCM    │  - Vector storage (usearch) │
└─────────────────────┘         └─────────────────────────────┘
```

## Features

- **Streaming ingestion**: Stream audio frames during playback for embedding extraction
- **Semantic search**: Find tracks by natural language queries ("upbeat electronic for working out")
- **Similar tracks**: Find tracks similar to a given track based on audio embeddings

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Rust | Single binary, no GIL, minimal resources |
| ML Runtime | ort (ONNX Runtime) | CUDA support, 3-5x faster than Python |
| Vector DB | usearch | Lightweight, embedded vector search |
| HTTP Server | axum | Async, high performance |
| Serialization | rmp-serde | Efficient MessagePack transport |
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

- HTTP with MessagePack bodies (Content-Type: `application/msgpack`)
- Binary PCM data for audio frames (Content-Type: `application/octet-stream`)

### Endpoints

#### Streaming Ingestion

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/stream/start` | Start ingestion session for a track |
| `POST` | `/api/v1/stream/{session_id}/frames` | Send PCM audio frames |
| `POST` | `/api/v1/stream/{session_id}/end` | End session and store embeddings |
| `GET` | `/api/v1/stream/{session_id}/status` | Check session status |
| `DELETE` | `/api/v1/stream/{session_id}` | Abort/cancel session |

#### Tracks

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/tracks/{track_id}` | Get track info (with optional embeddings) |
| `DELETE` | `/api/v1/tracks/{track_id}` | Remove track |
| `POST` | `/api/v1/tracks/upsert` | Upsert track with pre-computed embeddings |
| `POST` | `/api/v1/tracks/batch-upsert` | Batch upsert multiple tracks |

#### Embedding & Search

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/embed/text` | Generate text embedding from text or metadata |
| `POST` | `/api/v1/embed/audio` | Generate audio embedding from PCM data |
| `POST` | `/api/v1/tracks/search` | Search tracks by embedding vector |
| `POST` | `/api/v1/tracks/embed-text` | Generate text embedding and store in one call |
| `POST` | `/api/v1/tracks/batch-embed-text` | Batch generate and store text embeddings |

#### Mood Classification

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/mood/classify` | Classify mood from embedding or track ID |
| `GET` | `/api/v1/mood/list` | List available mood categories |

#### Model Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/models` | List available models |
| `POST` | `/api/v1/models/download` | Download a model |
| `GET` | `/api/v1/models/downloads` | List active downloads |
| `POST` | `/api/v1/models/{model_id}/load` | Load a model |
| `DELETE` | `/api/v1/models/{model_id}` | Delete a cached model |

#### Admin

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/config` | Get current configuration |
| `GET` | `/api/v1/status` | Get system status |
| `GET` | `/api/v1/storage/stats` | Get storage statistics |

### Data Types

```rust
// Start stream request
struct StartStreamRequest {
    track_id: String,
    metadata: TrackMetadata,
    format: AudioFormat,      // "pcm_f32_le", "pcm_s16_le", "pcm_s24_le"
    sample_rate: u32,         // 44100, 48000, etc.
    channels: u8,             // 1 (mono) or 2 (stereo)
}

// Track metadata
struct TrackMetadata {
    name: String,
    artists: Vec<String>,
    album: Option<String>,
    genres: Vec<String>,
}

// End stream request
struct EndStreamRequest {
    store: bool,              // Whether to store the embeddings
    min_duration_s: f32,      // Minimum duration to generate embedding (default: 3.0)
}

// Search request
struct SearchRequest {
    embedding: Vec<f32>,      // 512-dimensional embedding vector
    collection: String,       // "text" or "audio"
    limit: u32,               // Max results to return
    filter: Option<SearchFilter>,
}

struct SearchFilter {
    exclude_ids: Vec<String>, // Track IDs to exclude from results
}
```

## Configuration

Environment variables (use `INSIGHT_` prefix, double underscore for nesting):

```bash
# Model
INSIGHT_MODEL__NAME=Xenova/clap-htsat-unfused
INSIGHT_MODEL__ENABLE_CUDA=true         # NVIDIA GPU
INSIGHT_MODEL__ENABLE_ROCM=false        # AMD GPU
INSIGHT_MODEL__ENABLE_COREML=false      # Apple Silicon
INSIGHT_MODEL__ENABLE_DIRECTML=false    # Windows GPU
INSIGHT_MODEL__ENABLE_OPENVINO=false    # Intel acceleration

# Audio processing
INSIGHT_AUDIO__WINDOW_SIZE_S=10.0       # Window size for embeddings
INSIGHT_AUDIO__HOP_SIZE_S=10.0          # Hop between windows

# Storage
INSIGHT_STORAGE__MODE=file              # "file" (usearch) or "qdrant"
INSIGHT_STORAGE__DATA_DIR=~/.local/share/insight-sidecar
INSIGHT_STORAGE__URL=http://localhost:6334  # For Qdrant mode
INSIGHT_STORAGE__API_KEY=               # Qdrant Cloud API key
INSIGHT_STORAGE__ENABLED=true

# Server
INSIGHT_SERVER__HOST=0.0.0.0
INSIGHT_SERVER__PORT=8096
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
│   │   └── stream.rs     # Streaming session handler
│   ├── inference/
│   │   ├── mod.rs
│   │   ├── clap.rs       # CLAP model wrapper
│   │   └── audio.rs      # Audio preprocessing
│   └── storage/
│       ├── mod.rs
│       └── vector.rs     # Vector storage (usearch)
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
- [usearch](https://github.com/unum-cloud/usearch) - Vector search engine
