# AGENTS.md

Instructions for AI agents (Claude, Copilot, etc.) working with this codebase.

## Project Context

This is a Rust sidecar service for Music Assistant that provides ML inference for audio/text embeddings. It is intentionally separate from the main Python-based Music Assistant server to isolate heavy ML dependencies.

## Tech Stack

| Component | Crate | Purpose |
|-----------|-------|---------|
| HTTP Server | `axum` | Async web framework |
| Async Runtime | `tokio` | Async runtime |
| ML Inference | `ort` | ONNX Runtime bindings |
| Vector Storage | `usearch` / `qdrant-client` | File-based or hosted vector DB |
| Serialization | `rmp-serde` | MessagePack for all API endpoints |
| Audio Decoding | `symphonia` | Pure Rust audio decoding |
| Audio Resampling | `rubato` | High-quality resampling to 48kHz |
| Config | `config` | Environment/file config |
| Logging | `tracing` | Structured logging |

## Code Conventions

### Rust Style

- Follow standard Rust conventions (rustfmt, clippy)
- Use `thiserror` for error types
- Use `anyhow` for application-level error handling
- Prefer `impl Trait` over `dyn Trait` where possible
- Use `#[must_use]` on functions returning values that shouldn't be ignored

### Async Patterns

```rust
// Prefer async functions over manual Future implementations
async fn process_track(&self, track: Track) -> Result<Embedding> {
    // ...
}

// Use tokio::spawn for background tasks
tokio::spawn(async move {
    // background work
});

// Use channels for cross-task communication
let (tx, rx) = tokio::sync::mpsc::channel(32);
```

### Error Handling

```rust
// Define domain-specific errors
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Invalid audio format: {0}")]
    InvalidAudioFormat(String),

    #[error("ONNX runtime error: {0}")]
    OnnxError(#[from] ort::Error),
}

// Use anyhow::Result at application boundaries
pub async fn main() -> anyhow::Result<()> {
    // ...
}
```

### API Handlers

```rust
// Use axum extractors
async fn upsert_track(
    State(state): State<AppState>,
    MsgPack(payload): MsgPack<UpsertRequest>,
) -> Result<MsgPack<UpsertResponse>, ApiError> {
    // ...
}

// Return proper error responses
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = rmp_serde::to_vec(&ErrorResponse {
            error: ErrorDetail {
                code: self.code(),
                message: self.to_string(),
            },
        }).unwrap();

        (self.status_code(), body).into_response()
    }
}
```

## Project Structure

```
src/
├── main.rs              # Entry point, server setup
├── config.rs            # Configuration from env/files
├── error.rs             # Error types
├── server/
│   ├── mod.rs           # Router setup, AppState
│   ├── routes.rs        # Route definitions & MsgPack response type
│   ├── stream.rs        # Streaming session handler (real-time ingestion)
│   ├── tracks.rs        # Track CRUD & search handlers
│   ├── embed.rs         # Embedding generation handlers
│   ├── mood.rs          # Mood classification handlers
│   ├── models.rs        # Model management handlers
│   ├── extractors.rs    # Custom axum extractors (MsgPackExtractor)
│   └── watcher.rs       # Folder watcher API handlers
├── inference/
│   ├── mod.rs           # Inference orchestration, ClapModel
│   ├── clap.rs          # CLAP model wrapper (ONNX)
│   ├── audio.rs         # Audio preprocessing, AudioData types
│   └── mood.rs          # Mood classification
├── storage/
│   ├── mod.rs           # Storage trait, TrackMetadata
│   ├── file.rs          # File-based storage (usearch)
│   └── qdrant.rs        # Qdrant implementation
├── watcher/             # Optional folder watcher feature
│   ├── mod.rs           # Watcher orchestration
│   └── processor.rs     # File processing
└── types/
    ├── mod.rs           # Re-exports
    └── api.rs           # API request/response types
```

## Key Implementation Notes

### ONNX Model Loading

```rust
// Models are loaded once at startup
// Use Arc for sharing across handlers
pub struct ClapModel {
    session: ort::Session,
    // Embedding dimension (512 for CLAP)
    embedding_dim: usize,
}

impl ClapModel {
    pub fn load(model_path: &Path, use_cuda: bool) -> Result<Self> {
        let builder = ort::Session::builder()?;

        if use_cuda {
            builder.with_execution_providers([
                ort::CUDAExecutionProvider::default().build()
            ])?;
        }

        let session = builder.commit_from_file(model_path)?;
        Ok(Self { session, embedding_dim: 512 })
    }
}
```

### Audio Processing

- CLAP expects mono audio at 48kHz
- Use sliding windows (default 10s) for long audio
- Average embeddings across windows for final representation

```rust
// Resample and convert to mono if needed
fn preprocess_audio(
    samples: &[f32],
    sample_rate: u32,
    channels: u8,
) -> Vec<f32> {
    // Convert to mono
    let mono = if channels == 2 {
        samples.chunks(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect()
    } else {
        samples.to_vec()
    };

    // Resample to 48kHz if needed
    if sample_rate != 48000 {
        resample(&mono, sample_rate, 48000)
    } else {
        mono
    }
}
```

### Vector Storage

Two storage backends available (configurable via `INSIGHT_STORAGE__MODE`):

1. **File-based (usearch)** - Default, embedded HNSW index
2. **Qdrant** - Hosted vector database for production

Both store text and audio embeddings in separate collections:
- `tracks_text` - Text embeddings from metadata
- `tracks_audio` - Audio embeddings from PCM

```rust
// Storage trait allows backend switching
pub trait VectorStorage: Send + Sync {
    async fn upsert(&self, collection: &str, id: &str, embedding: &[f32], metadata: TrackMetadata) -> Result<()>;
    async fn search(&self, collection: &str, embedding: &[f32], limit: usize, filter: Option<SearchFilter>) -> Result<Vec<SearchResult>>;
    async fn get(&self, collection: &str, id: &str) -> Result<Option<StoredTrack>>;
    async fn delete(&self, collection: &str, id: &str) -> Result<bool>;
}
```

### Streaming Ingestion

The streaming API allows real-time ingestion during playback:

```rust
// Session-based streaming in stream.rs
pub struct StreamSession {
    id: Uuid,
    track_id: String,
    metadata: IngestMetadata,
    format: AudioFormat,
    source_sample_rate: u32,
    channels: u8,

    // Buffers - audio is resampled to 48kHz mono
    mono_buffer: Vec<f32>,           // Pre-resampling
    resampled_buffer: Vec<f32>,      // Post-resampling
    resampler: Option<FftFixedIn<f32>>,

    // Embeddings from 10-second windows
    window_embeddings: Vec<Vec<f32>>,
    status: StreamSessionStatus,
}
```

Flow:
1. `POST /stream/start` - Creates session, returns session_id
2. `POST /stream/{id}/frames` - Send PCM bytes, buffers & generates embeddings
3. `POST /stream/{id}/end` - Finalizes, averages embeddings, stores to vector DB

### MessagePack API

- All request/response bodies use msgpack (including streaming endpoints)
- Embeddings are sent as raw bytes (not base64)
- Use `rmp_serde` with struct flattening disabled

```rust
// Custom extractor for msgpack
pub struct MsgPack<T>(pub T);

#[async_trait]
impl<T, S> FromRequest<S> for MsgPack<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let bytes = Bytes::from_request(req, state).await?;
        let value = rmp_serde::from_slice(&bytes)?;
        Ok(MsgPack(value))
    }
}
```

## Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_preprocessing() {
        let stereo = vec![1.0, 0.5, 0.8, 0.2];
        let mono = to_mono(&stereo);
        assert_eq!(mono, vec![0.75, 0.5]);
    }
}
```

### Integration Tests

```rust
// tests/api_test.rs
#[tokio::test]
async fn test_upsert_and_search() {
    let app = create_test_app().await;

    // Upsert a track
    let response = app
        .post("/api/v1/tracks/upsert")
        .msgpack(&UpsertRequest { /* ... */ })
        .await;

    assert_eq!(response.status(), StatusCode::OK);

    // Search for it
    let response = app
        .post("/api/v1/search")
        .msgpack(&SearchRequest { query: "rock music".into(), /* ... */ })
        .await;

    let results: SearchResponse = response.msgpack().await;
    assert!(!results.results.is_empty());
}
```

## Common Tasks

### Adding a New Endpoint

1. Define request/response types in `src/types/`
2. Add handler in `src/server/routes.rs`
3. Register route in `src/server/mod.rs`
4. Add tests in `tests/`

### Adding a New Model

1. Export model to ONNX format
2. Add model config variant in `src/config.rs`
3. Update model loading in `src/inference/clap.rs`
4. Test with integration tests

### Debugging Inference Issues

- Enable `RUST_LOG=ort=debug` for ONNX Runtime logs
- Check input tensor shapes match model expectations
- Verify audio preprocessing (sample rate, channels, normalization)

## Performance Considerations

- Model loading is slow (~3-8s) - do it once at startup
- Use `Arc<ClapModel>` to share across handlers
- Batch inference when possible (batch upsert endpoint)
- Audio preprocessing can be parallelized with `rayon`
- Qdrant queries are fast, but batch deletes can be slow

## Do Not

- Do not use Python or call Python from Rust
- Do not add unnecessary dependencies
- Do not block the async runtime with CPU-intensive work (use `spawn_blocking`)
- Do not store embeddings as base64 in the API (use raw bytes)
- Do not hardcode model paths or configuration values
