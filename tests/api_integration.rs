//! Integration tests for API endpoints.
//!
//! These tests verify the API endpoints work correctly without requiring
//! a real ML model (which would be too slow for unit tests).
//!
//! Tests cover:
//! - Health and config endpoints with response deserialization
//! - Model management endpoints
//! - Storage endpoints
//! - Streaming endpoints (without model)
//! - Error handling for missing resources

use axum_test::TestServer;
use bytes::Bytes;
use insight_sidecar::config::AppConfig;
use insight_sidecar::server::{create_router, AppState};
use insight_sidecar::types::models::StorageStatsResponse;
use insight_sidecar::types::{ConfigResponse, HealthResponse, HealthStatus};

/// Create a test server with default configuration (no model loaded)
fn create_test_server() -> TestServer {
    let config = AppConfig::default();
    let state = AppState::new(config);
    let app = create_router(state);
    TestServer::new(app).unwrap()
}

/// Helper to create msgpack bytes from named fields
fn msgpack_bytes<T: serde::Serialize>(value: &T) -> Bytes {
    Bytes::from(rmp_serde::to_vec_named(value).unwrap())
}

/// Deserialize msgpack response body
fn from_msgpack<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> T {
    rmp_serde::from_slice(bytes).expect("Failed to deserialize msgpack response")
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

    // Deserialize and verify response structure
    let health: HealthResponse = from_msgpack(response.as_bytes());
    assert!(!health.version.is_empty());
    // Without model loaded, status should be degraded (with inference feature)
    // or healthy (without inference feature)
    assert!(
        health.status == HealthStatus::Degraded || health.status == HealthStatus::Healthy,
        "Unexpected health status: {:?}",
        health.status
    );
    assert!(!health.model_loaded);
}

#[tokio::test]
async fn test_health_response_version_format() {
    let server = create_test_server();

    let response = server.get("/api/v1/health").await;
    response.assert_status_ok();

    let health: HealthResponse = from_msgpack(response.as_bytes());
    // Version should be semver format (e.g., "0.1.0")
    assert!(
        health.version.contains('.'),
        "Version should be semver format: {}",
        health.version
    );
}

#[tokio::test]
async fn test_config_endpoint() {
    let server = create_test_server();

    let response = server.get("/api/v1/config").await;

    response.assert_status_ok();

    // Deserialize and verify response structure
    let config: ConfigResponse = from_msgpack(response.as_bytes());
    assert!(!config.model.name.is_empty());
    assert!(config.audio.window_size_s > 0.0);
    assert!(config.audio.hop_size_s > 0.0);
    assert!(config.server.port > 0);
}

#[tokio::test]
async fn test_config_model_not_loaded() {
    let server = create_test_server();

    let response = server.get("/api/v1/config").await;
    response.assert_status_ok();

    let config: ConfigResponse = from_msgpack(response.as_bytes());
    // Model should not be loaded in test server
    assert!(!config.model.loaded);
    assert!(config.model.device.is_none());
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

    // Deserialize and verify response structure (wrapped in StorageStatsResponse)
    let response_body: StorageStatsResponse = from_msgpack(response.as_bytes());
    let stats = response_body.stats;
    // Without storage connected, counts should be 0
    assert_eq!(stats.text_collection_count, 0);
    assert_eq!(stats.audio_collection_count, 0);
    assert_eq!(stats.total_tracks, 0);
    // Mode should be present
    assert!(!stats.mode.is_empty());
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

// ============================================================================
// Streaming endpoint tests
// ============================================================================

#[tokio::test]
async fn test_stream_frames_invalid_session() {
    let server = create_test_server();

    // Try to send frames to a non-existent session
    let response = server
        .post("/api/v1/stream/550e8400-e29b-41d4-a716-446655440000/frames")
        .bytes(Bytes::from(vec![0u8; 100]))
        .await;

    // Should return not found
    response.assert_status_not_found();
}

#[tokio::test]
async fn test_stream_end_invalid_session() {
    let server = create_test_server();

    // Try to end a non-existent session
    let response = server
        .post("/api/v1/stream/550e8400-e29b-41d4-a716-446655440000/end")
        .content_type("application/msgpack")
        .bytes(msgpack_bytes(&serde_json::json!({
            "store": false
        })))
        .await;

    // Returns 503 (model not loaded) rather than 404 because model check happens first
    // This is expected behavior - model must be loaded to end a stream
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_stream_status_invalid_session() {
    let server = create_test_server();

    // Try to get status of non-existent session
    let response = server
        .get("/api/v1/stream/550e8400-e29b-41d4-a716-446655440000/status")
        .await;

    // Should return not found
    response.assert_status_not_found();
}

#[tokio::test]
async fn test_stream_abort_invalid_session() {
    let server = create_test_server();

    // Try to abort a non-existent session
    let response = server
        .delete("/api/v1/stream/550e8400-e29b-41d4-a716-446655440000")
        .await;

    // Should return not found
    response.assert_status_not_found();
}

#[tokio::test]
async fn test_stream_invalid_uuid_format() {
    let server = create_test_server();

    // Try to access with invalid UUID format
    let response = server
        .get("/api/v1/stream/not-a-valid-uuid/status")
        .await;

    // Should return bad request
    response.assert_status_bad_request();
}

// ============================================================================
// Tracks endpoint tests
// ============================================================================

#[tokio::test]
async fn test_tracks_similar_without_storage() {
    let server = create_test_server();

    let response = server
        .post("/api/v1/tracks/similar")
        .content_type("application/msgpack")
        .bytes(msgpack_bytes(&serde_json::json!({
            "track_id": "test_track_123",
            "collection": "text",
            "limit": 10
        })))
        .await;

    // Should fail because storage is not configured
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_tracks_get_without_storage() {
    let server = create_test_server();

    let response = server.get("/api/v1/tracks/test_track_123").await;

    // Should fail because storage is not configured
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_tracks_delete_without_storage() {
    let server = create_test_server();

    let response = server.delete("/api/v1/tracks/test_track_123").await;

    // Should fail because storage is not configured
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_batch_upsert_without_storage() {
    let server = create_test_server();

    let response = server
        .post("/api/v1/tracks/batch-upsert")
        .content_type("application/msgpack")
        .bytes(msgpack_bytes(&serde_json::json!({
            "tracks": []
        })))
        .await;

    // Should fail because storage is not configured
    response.assert_status_not_ok();
}

// ============================================================================
// Taste profile endpoint tests
// ============================================================================

#[tokio::test]
async fn test_taste_compute_without_storage() {
    let server = create_test_server();

    let response = server
        .post("/api/v1/taste/compute")
        .content_type("application/msgpack")
        .bytes(msgpack_bytes(&serde_json::json!({
            "user_id": "test_user",
            "interactions": []
        })))
        .await;

    // Should fail because storage is not configured
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_taste_recommend_without_storage() {
    let server = create_test_server();

    let response = server
        .post("/api/v1/taste/recommend")
        .content_type("application/msgpack")
        .bytes(msgpack_bytes(&serde_json::json!({
            "user_id": "test_user",
            "limit": 10
        })))
        .await;

    // Should fail because storage is not configured
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_taste_profile_get_without_storage() {
    let server = create_test_server();

    let response = server.get("/api/v1/taste/test_user/global").await;

    // Should fail because storage is not configured
    response.assert_status_not_ok();
}

// ============================================================================
// Content-type validation tests
// ============================================================================

#[tokio::test]
async fn test_embed_text_wrong_content_type() {
    let server = create_test_server();

    // Send JSON instead of msgpack
    let response = server
        .post("/api/v1/embed/text")
        .content_type("application/json")
        .json(&serde_json::json!({
            "text": "test song"
        }))
        .await;

    // Should fail (expects msgpack) or fail because no model
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_msgpack_content_type_header() {
    let server = create_test_server();

    let response = server.get("/api/v1/health").await;
    response.assert_status_ok();

    let content_type = response.headers().get("content-type").unwrap();
    assert_eq!(content_type.to_str().unwrap(), "application/msgpack");
}

// ============================================================================
// Edge case tests
// ============================================================================

#[tokio::test]
async fn test_empty_batch_request() {
    let server = create_test_server();

    // Empty batch should still be valid syntax but fail due to no storage
    let response = server
        .post("/api/v1/tracks/batch-embed-text")
        .content_type("application/msgpack")
        .bytes(msgpack_bytes(&serde_json::json!({
            "tracks": []
        })))
        .await;

    // Fails because no model loaded (but request format is valid)
    response.assert_status_not_ok();
}

#[tokio::test]
async fn test_unknown_endpoint_returns_404() {
    let server = create_test_server();

    let response = server.get("/api/v1/nonexistent/endpoint").await;

    response.assert_status_not_found();
}

#[tokio::test]
async fn test_root_endpoint_not_available() {
    let server = create_test_server();

    let response = server.get("/").await;

    // Root is not mapped, should return 404
    response.assert_status_not_found();
}
