//! Custom extractors for the HTTP server.

use axum::{
    async_trait,
    body::Bytes,
    extract::{FromRequest, Request},
    http::{header::CONTENT_TYPE, StatusCode},
    response::{IntoResponse, Response},
};
use serde::de::DeserializeOwned;

/// Rejection type for `MsgPackExtractor`
pub struct MsgPackRejection {
    message: String,
}

impl IntoResponse for MsgPackRejection {
    fn into_response(self) -> Response {
        let body = crate::error::ErrorResponse {
            error: crate::error::ErrorDetail {
                code: "DESERIALIZATION_ERROR",
                message: self.message.clone(),
            },
        };

        match rmp_serde::to_vec_named(&body) {
            Ok(bytes) => (
                StatusCode::BAD_REQUEST,
                [("content-type", "application/msgpack")],
                bytes,
            )
                .into_response(),
            Err(_) => (StatusCode::BAD_REQUEST, self.message).into_response(),
        }
    }
}

/// Extractor for `MessagePack` request bodies.
///
/// This extractor deserializes the request body from `MessagePack` format.
/// It accepts both `application/msgpack` and `application/x-msgpack` content types.
pub struct MsgPackExtractor<T>(pub T);

#[async_trait]
impl<T, S> FromRequest<S> for MsgPackExtractor<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = MsgPackRejection;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        // Check content type
        let content_type = req
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if !content_type.contains("msgpack") && !content_type.is_empty() {
            return Err(MsgPackRejection {
                message: format!(
                    "Invalid content type: expected application/msgpack, got {content_type}"
                ),
            });
        }

        // Extract body bytes
        let bytes = Bytes::from_request(req, state)
            .await
            .map_err(|e| MsgPackRejection {
                message: format!("Failed to read request body: {e}"),
            })?;

        // Deserialize from MessagePack
        rmp_serde::from_slice(&bytes)
            .map(MsgPackExtractor)
            .map_err(|e| MsgPackRejection {
                message: format!("Failed to deserialize MessagePack: {e}"),
            })
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestPayload {
        name: String,
        value: i32,
    }

    #[test]
    fn test_msgpack_roundtrip() {
        let payload = TestPayload {
            name: "test".to_string(),
            value: 42,
        };

        let bytes = rmp_serde::to_vec(&payload).unwrap();
        let decoded: TestPayload = rmp_serde::from_slice(&bytes).unwrap();

        assert_eq!(payload, decoded);
    }
}
