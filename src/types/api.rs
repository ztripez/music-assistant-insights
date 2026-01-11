//! API request and response types.
//!
//! This module re-exports all API types from their domain-specific modules
//! for backward compatibility.

// Re-export all domain-specific types
pub use super::embed::*;
pub use super::models::*;
pub use super::mood::*;
pub use super::taste::*;
pub use super::tracks::*;
