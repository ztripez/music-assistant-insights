use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use thiserror::Error;

/// Application-level errors
#[derive(Debug, Error)]
pub enum AppError {
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("Internal server error: {0}")]
    Internal(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl AppError {
    /// Returns the appropriate HTTP status code for this error
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::Config(_) | Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::BadRequest(_) | Self::Serialization(_) => StatusCode::BAD_REQUEST,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
        }
    }

    /// Returns a machine-readable error code
    pub fn code(&self) -> &'static str {
        match self {
            Self::Config(_) => "CONFIG_ERROR",
            Self::Internal(_) => "INTERNAL_ERROR",
            Self::BadRequest(_) => "BAD_REQUEST",
            Self::NotFound(_) => "NOT_FOUND",
            Self::Serialization(_) => "SERIALIZATION_ERROR",
        }
    }
}

/// Error response body structure
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub code: &'static str,
    pub message: String,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = ErrorResponse {
            error: ErrorDetail {
                code: self.code(),
                message: self.to_string(),
            },
        };

        // Try to serialize as msgpack, fall back to JSON
        if let Ok(bytes) = rmp_serde::to_vec_named(&body) {
            (status, [("content-type", "application/msgpack")], bytes).into_response()
        } else {
            // Fallback to JSON if msgpack fails
            let json = serde_json::to_string(&body).unwrap_or_else(|_| {
                r#"{"error":{"code":"SERIALIZATION_ERROR","message":"Failed to serialize error"}}"#.to_string()
            });
            (status, [("content-type", "application/json")], json).into_response()
        }
    }
}

/// Result type alias for handlers
pub type Result<T> = std::result::Result<T, AppError>;
