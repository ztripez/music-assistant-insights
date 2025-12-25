//! Integration tests for API endpoints.
//!
//! These tests verify the API endpoints work correctly without requiring
//! a real ML model (which would be too slow for unit tests).

use axum_test::TestServer;
use bytes::Bytes;
use insight_sidecar::config::AppConfig;
use insight_sidecar::server::{create_router, AppState};

/// Create a test server with default configuration (no model loaded)
fn create_test_server() -> TestServer {
    let config = AppConfig::default();
    let state = AppState::new(config);
    let app = create_router(state);
    TestServer::new(app).unwrap()
}

/// Helper to create msgpack bytes
fn msgpack_bytes<T: serde::Serialize>(value: &T) -> Bytes {
    Bytes::from(rmp_serde::to_vec(value).unwrap())
}

#[tokio::test]
async fn test_health_endpoint() {
    let server = create_test_server();

    let response = server.get("/api/v1/health").await;

    response.assert_status_ok();
    // Response is msgpack - check content-type
    let content_type = response.headers().get("content-type");
    assert!(content_type.is_some());
    assert!(content_type.unwrap().to_str().unwrap().contains("msgpack"));
}

#[tokio::test]
async fn test_config_endpoint() {
    let server = create_test_server();

    let response = server.get("/api/v1/config").await;

    response.assert_status_ok();
}

#[tokio::test]
async fn test_status_endpoint() {
    let server = create_test_server();

    let response = server.get("/api/v1/status").await;

    response.assert_status_ok();
    let body = response.text();
    // Should contain version info
    assert!(body.contains("version") || body.contains("uptime"));
}

#[tokio::test]
#[cfg(feature = "inference")]
async fn test_models_list_endpoint() {
    let server = create_test_server();

    let response = server.get("/api/v1/models").await;

    response.assert_status_ok();
    // Response is msgpack - check content-type
    let content_type = response.headers().get("content-type");
    assert!(content_type.is_some());
    assert!(content_type.unwrap().to_str().unwrap().contains("msgpack"));
}

#[tokio::test]
#[cfg(not(feature = "inference"))]
async fn test_models_list_endpoint() {
    let server = create_test_server();

    let response = server.get("/api/v1/models").await;

    // Without inference feature, models endpoint returns error
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_mood_list_endpoint() {
    let server = create_test_server();

    let response = server.get("/api/v1/mood/list").await;

    response.assert_status_ok();
    let body = response.text();
    // Should contain moods
    assert!(body.contains("moods") || body.contains("energetic"));
}

#[tokio::test]
async fn test_storage_stats_endpoint() {
    let server = create_test_server();

    let response = server.get("/api/v1/storage/stats").await;

    // Storage stats endpoint exists (returns error if storage not initialized)
    // With default config, storage is enabled but not connected, so returns 200 with zeros
    response.assert_status_ok();
    let content_type = response.headers().get("content-type");
    assert!(content_type.is_some());
}

#[tokio::test]
async fn test_embed_text_without_model() {
    let server = create_test_server();

    // Try to embed text without a model loaded
    let response = server
        .post("/api/v1/embed/text")
        .content_type("application/msgpack")
        .bytes(msgpack_bytes(&serde_json::json!({
            "text": "test song"
        })))
        .await;

    // Should fail because no model is loaded
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_stream_start_without_model() {
    let server = create_test_server();

    // Try to start streaming without a model loaded
    let response = server
        .post("/api/v1/stream/start")
        .json(&serde_json::json!({
            "track_id": "test_123",
            "metadata": {
                "name": "Test Song",
                "artists": ["Artist"]
            },
            "format": "pcm_s16_le",
            "sample_rate": 44100,
            "channels": 2
        }))
        .await;

    // Should fail because no model is loaded
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_search_without_storage() {
    let server = create_test_server();

    let response = server
        .post("/api/v1/tracks/search")
        .content_type("application/msgpack")
        .bytes(msgpack_bytes(&serde_json::json!({
            "embedding": vec![0.1f32; 512],
            "collection": "text",
            "limit": 10
        })))
        .await;

    // Should fail because storage is not configured
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_mood_classify_without_model() {
    let server = create_test_server();

    let response = server
        .post("/api/v1/mood/classify")
        .content_type("application/msgpack")
        .bytes(msgpack_bytes(&serde_json::json!({
            "embedding": vec![0.1f32; 512]
        })))
        .await;

    // Should fail because no model is loaded
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_delete_nonexistent_model() {
    let server = create_test_server();

    let response = server.delete("/api/v1/models/nonexistent/model").await;

    // Should return not found or error
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_load_nonexistent_model() {
    let server = create_test_server();

    let response = server.post("/api/v1/models/nonexistent%2Fmodel/load").await;

    // Should return error (model not downloaded)
    response.assert_status_not_ok();
}
