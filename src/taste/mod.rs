//! Taste profile computation module.
//!
//! This module computes user taste vectors from listening history using
//! weighted embedding aggregation.

mod compute;

pub use compute::{TasteVectorComputer, TasteVectorError};
