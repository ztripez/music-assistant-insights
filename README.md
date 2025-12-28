# Music Assistant Insight Sidecar

A lightweight, high-performance inference sidecar for Music Assistant that provides audio and text embeddings using CLAP (Contrastive Language-Audio Pretraining) models, with mood classification, taste profile computation, and personalized recommendations.

## Overview

This sidecar offloads ML inference from Music Assistant, keeping the main server free of heavy ML/CUDA dependencies. It provides:

- **Text embeddings** from track metadata (artist, album, genre)
- **Audio embeddings** from PCM audio streamed during playback
- **Mood classification** using zero-shot CLAP inference
- **Taste profiles** computed from user listening history
- **Personalized recommendations** based on taste profiles
- **Vector storage** with Qdrant or embedded usearch

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Music Assistant                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Music Insights Provider                      │   │
│  │  - Subscribes to library events (add/update/delete)      │   │
│  │  - Subscribes to playback events (MEDIA_ITEM_PLAYED)     │   │
│  │  - Streams audio during playback                         │   │
│  │  - Computes taste profiles from listening history        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              │ HTTP (MessagePack)                │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Insight Sidecar (Rust)                         │   │
│  │  - CLAP model inference (ONNX Runtime)                   │   │
│  │  - Text embeddings from track metadata                   │   │
│  │  - Audio embeddings from streaming audio                 │   │
│  │  - Mood classification (valence/arousal)                 │   │
│  │  - Taste profile computation                             │   │
│  │  - Vector similarity search                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Vector Storage                               │   │
│  │  - Qdrant (remote/docker) OR                             │   │
│  │  - usearch (local file-based)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Core Features

- **Streaming ingestion**: Stream audio frames during playback for real-time embedding extraction
- **Semantic search**: Find tracks by natural language queries ("upbeat electronic for working out")
- **Similar tracks**: Find tracks similar to a given track based on embeddings
- **Mood classification**: Zero-shot mood classification with valence/arousal coordinates
- **Taste profiles**: Compute user preference vectors from listening history
- **Personalized recommendations**: Get track recommendations based on taste profiles

### Folder Watcher (Optional)

When built with the `watcher` feature, the sidecar can monitor local music directories:

- Watches configured folders for audio files (mp3, flac, ogg, m4a, etc.)
- Decodes audio using symphonia (pure Rust)
- Extracts ID3/Vorbis metadata
- Generates embeddings automatically
- Useful for pre-populating the vector DB or standalone usage

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Rust | Single binary, no GIL, minimal resources |
| ML Runtime | ort (ONNX Runtime) | CUDA support, 3-5x faster than Python |
| Vector DB | Qdrant / usearch | Qdrant for production, usearch for embedded |
| HTTP Server | axum | Async, high performance |
| Serialization | rmp-serde | Efficient MessagePack transport |
| Audio | symphonia + rubato | Pure Rust decoding and resampling |
| Mel Spectrogram | mel-spec | STFT and filterbank computation |

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

#### Health & Configuration

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/config` | Get current configuration |
| `GET` | `/api/v1/status` | Full system status |

#### Streaming Ingestion

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/stream/start` | Start ingestion session for a track |
| `POST` | `/api/v1/stream/{session_id}/frames` | Send PCM audio frames (raw bytes) |
| `POST` | `/api/v1/stream/{session_id}/end` | End session and store embeddings |
| `GET` | `/api/v1/stream/{session_id}/status` | Check session status |
| `DELETE` | `/api/v1/stream/{session_id}` | Abort/cancel session |

#### Embedding Generation

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/embed/text` | Generate text embedding from text or metadata |
| `POST` | `/api/v1/embed/audio` | Generate audio embedding from PCM data |

#### Track Storage

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/tracks/{track_id}` | Get track info and embeddings |
| `DELETE` | `/api/v1/tracks/{track_id}` | Remove track from storage |
| `POST` | `/api/v1/tracks/upsert` | Upsert track with pre-computed embeddings |
| `POST` | `/api/v1/tracks/batch-upsert` | Batch upsert multiple tracks |
| `POST` | `/api/v1/tracks/embed-text` | Generate text embedding and store |
| `POST` | `/api/v1/tracks/batch-embed-text` | Batch generate and store text embeddings |
| `POST` | `/api/v1/tracks/search` | Search by embedding vector |
| `POST` | `/api/v1/tracks/search-text` | Search by natural language query |

#### Mood Classification

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/mood/classify` | Classify mood from embedding or track ID |
| `GET` | `/api/v1/mood/list` | List available mood categories |

#### Taste Profiles & Recommendations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/users/{user_id}/profile/compute` | Compute taste profile from interactions |
| `POST` | `/api/v1/users/{user_id}/recommend` | Get personalized recommendations |
| `GET` | `/api/v1/users/{user_id}/profile/vector` | Get taste vector (for debugging) |
| `DELETE` | `/api/v1/users/{user_id}/profile` | Delete a specific profile |
| `DELETE` | `/api/v1/users/{user_id}/profiles` | Delete all profiles for user |

#### Model Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/models` | List available models |
| `POST` | `/api/v1/models/download` | Start model download |
| `GET` | `/api/v1/models/downloads` | List active downloads |
| `POST` | `/api/v1/models/{model_id}/load` | Load a model |
| `DELETE` | `/api/v1/models/{model_id}` | Delete a cached model |

#### Storage

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/storage/stats` | Get storage statistics |

#### Folder Watcher (with `watcher` feature)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/watcher/status` | Get watcher status |
| `POST` | `/api/v1/watcher/start` | Start watching folders |
| `POST` | `/api/v1/watcher/stop` | Stop watching |
| `POST` | `/api/v1/watcher/pause` | Pause processing |
| `POST` | `/api/v1/watcher/resume` | Resume processing |
| `POST` | `/api/v1/watcher/scan` | Trigger immediate scan |
| `GET` | `/api/v1/watcher/folders` | List watched folders |
| `POST` | `/api/v1/watcher/folders` | Add folder to watch |
| `DELETE` | `/api/v1/watcher/folders/{path}` | Remove folder |

### Data Types

```rust
// Start stream request
struct StartStreamRequest {
    track_id: String,
    metadata: IngestMetadata,
    format: AudioFormat,      // "pcm_f32_le", "pcm_s16_le", "pcm_s24_le"
    sample_rate: u32,         // 44100, 48000, etc.
    channels: u8,             // 1 (mono) or 2 (stereo)
    replace_existing: bool,   // Replace existing session for track (default: true)
}

// Track metadata for ingestion
struct IngestMetadata {
    name: String,
    artists: Vec<String>,
    album: Option<String>,
    genres: Vec<String>,
}

// End stream request
struct EndStreamRequest {
    store: bool,              // Whether to store the embeddings (default: true)
    min_duration_s: f32,      // Minimum duration to generate embedding (default: 3.0)
}

// Mood classification result
struct MoodClassification {
    primary_mood: String,     // e.g., "energetic", "melancholic"
    moods: Vec<String>,       // Top moods
    mood_scores: HashMap<String, f32>,
    valence: f32,             // -1.0 to 1.0
    arousal: f32,             // -1.0 to 1.0
}

// User interaction for taste profile
struct UserInteraction {
    track_id: String,
    play_count: u32,
    skip_count: u32,
    is_favorite: bool,
    last_played: i64,         // Unix timestamp
}

// Search filter
struct SearchFilter {
    artists: Option<Vec<String>>,
    genres: Option<Vec<String>>,
    album: Option<String>,
    exclude_ids: Option<Vec<String>>,
    moods: Option<Vec<String>>,
    exclude_moods: Option<Vec<String>>,
    min_valence: Option<f32>,
    max_valence: Option<f32>,
    min_arousal: Option<f32>,
    max_arousal: Option<f32>,
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
INSIGHT_AUDIO__HOP_SIZE_S=10.0          # Hop between windows (50% overlap uses 5.0)

# Storage
INSIGHT_STORAGE__MODE=qdrant            # "file" (usearch) or "qdrant"
INSIGHT_STORAGE__DATA_DIR=~/.local/share/insight-sidecar
INSIGHT_STORAGE__URL=http://localhost:6334  # For Qdrant mode
INSIGHT_STORAGE__API_KEY=               # Qdrant Cloud API key
INSIGHT_STORAGE__COLLECTION_PREFIX=     # Optional prefix for collections
INSIGHT_STORAGE__ENABLED=true

# Server
INSIGHT_SERVER__HOST=0.0.0.0
INSIGHT_SERVER__PORT=8096
```

## Deployment

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
    environment:
      - INSIGHT_STORAGE__MODE=qdrant
      - INSIGHT_STORAGE__URL=http://qdrant:6334
    depends_on:
      - qdrant
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8096/api/v1/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant-data:/qdrant/storage

volumes:
  qdrant-data:
```

### Docker with CUDA

```yaml
services:
  insight-sidecar:
    image: ghcr.io/music-assistant/insight-sidecar:cuda
    environment:
      - INSIGHT_MODEL__ENABLE_CUDA=true
      - INSIGHT_STORAGE__MODE=file
    volumes:
      - insight-data:/data
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

## Development

### Prerequisites

- Rust 1.75+
- ONNX Runtime (automatically downloaded)
- (Optional) CUDA toolkit for GPU support

### Building

```bash
# Default (CPU, Qdrant storage)
cargo build --release

# With CUDA acceleration
cargo build --release --features cuda

# With folder watcher
cargo build --release --features watcher

# Full featured
cargo build --release --features "cuda,watcher"

# File-based storage only (no Qdrant dependency)
cargo build --release --features storage-file --no-default-features
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `inference` | CLAP model inference | Yes |
| `storage` | Qdrant vector storage | Yes |
| `storage-file` | usearch file-based storage | No |
| `cuda` | NVIDIA CUDA acceleration | No |
| `rocm` | AMD ROCm acceleration | No |
| `coreml` | Apple CoreML acceleration | No |
| `directml` | Windows DirectML acceleration | No |
| `openvino` | Intel OpenVINO acceleration | No |
| `watcher` | Folder watcher for local files | No |

### Testing

```bash
cargo test

# With all features
cargo test --all-features
```

### Project Structure

```
music-assistant-insights/
├── src/
│   ├── main.rs              # Entry point, CLI args
│   ├── config.rs            # Configuration loading
│   ├── error.rs             # Error types
│   ├── lib.rs               # Library exports
│   ├── server/
│   │   ├── mod.rs           # Router setup, AppState
│   │   ├── routes.rs        # Health, config endpoints
│   │   ├── embed.rs         # Embedding generation
│   │   ├── tracks.rs        # Track storage operations
│   │   ├── stream.rs        # Streaming session handler
│   │   ├── mood.rs          # Mood classification
│   │   ├── taste.rs         # Taste profile endpoints
│   │   ├── management.rs    # Model/storage management
│   │   ├── watcher.rs       # Folder watcher endpoints
│   │   └── extractors.rs    # Custom extractors (msgpack)
│   ├── inference/
│   │   ├── mod.rs           # ONNX model loading
│   │   ├── model.rs         # ClapModel wrapper
│   │   ├── audio.rs         # Audio preprocessing, mel spectrograms
│   │   ├── text.rs          # Text formatting for embeddings
│   │   ├── download.rs      # Model download manager
│   │   └── registry.rs      # Known models registry
│   ├── storage/
│   │   ├── mod.rs           # Storage trait and types
│   │   ├── qdrant.rs        # Qdrant backend
│   │   └── usearch_store.rs # usearch file backend
│   ├── mood/
│   │   ├── mod.rs           # MoodClassifier
│   │   └── prompts.rs       # Mood label definitions
│   ├── taste/
│   │   ├── mod.rs           # Taste profile types
│   │   └── compute.rs       # Profile computation logic
│   ├── watcher/             # Folder watcher (optional)
│   │   ├── mod.rs
│   │   ├── config.rs        # Watcher configuration
│   │   ├── service.rs       # Main watcher service
│   │   ├── scanner.rs       # Directory scanner
│   │   ├── watcher.rs       # File system watcher
│   │   ├── decoder.rs       # Audio file decoder (symphonia)
│   │   ├── metadata.rs      # ID3/Vorbis metadata extraction
│   │   ├── processor.rs     # Track processing pipeline
│   │   └── state.rs         # Watcher state tracking
│   └── types/
│       ├── mod.rs           # Shared types
│       └── api.rs           # API request/response types
├── Cargo.toml
└── README.md
```

## Collections

The sidecar maintains separate vector collections:

- `tracks_text` - Text embeddings from track metadata (512-dim)
- `tracks_audio` - Audio embeddings from streaming (512-dim)
- `taste_profiles` - User preference vectors (512-dim)

## Resource Requirements

| Configuration | CPU | Memory | Storage |
|---------------|-----|--------|---------|
| Minimal (CPU only) | 2 cores | 512MB | 500MB + 2KB/track |
| Recommended | 4 cores | 1GB | 500MB + 2KB/track |
| With CUDA | 4 cores + GPU | 2GB VRAM | 500MB + 2KB/track |

Inference latency:
- Text embedding: ~10-30ms (CPU), ~5-10ms (CUDA)
- Audio embedding (10s window): ~50-100ms (CPU), ~20-40ms (CUDA)
- Mood classification: ~5-15ms (uses cached prompt embeddings)

## License

Apache-2.0

## Related Projects

- [Music Assistant](https://github.com/music-assistant/server) - The main Music Assistant server
- [LAION CLAP](https://github.com/LAION-AI/CLAP) - The CLAP model architecture
- [Qdrant](https://github.com/qdrant/qdrant) - Vector search engine
- [usearch](https://github.com/unum-cloud/usearch) - Embedded vector search
