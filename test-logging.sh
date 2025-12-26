#!/bin/bash
# Test script for file logging feature
# Date: 2025-12-26
# Purpose: Demonstrate file logging capability for easy session sharing

set -e

LOG_DIR="./test-logs"
mkdir -p "$LOG_DIR"

echo "Starting sidecar with file logging enabled..."
echo "Log directory: $LOG_DIR"
echo ""
echo "Run with: --log-file --log-dir $LOG_DIR"
echo ""
echo "Example usage:"
echo "  ./target/debug/insight-sidecar --log-file"
echo "  ./target/debug/insight-sidecar --log-file --log-dir /custom/path"
echo "  INSIGHT_LOG_FILE=true ./target/debug/insight-sidecar"
echo ""
echo "Default log location: ~/.local/share/insight-sidecar/logs/insight-sidecar.log.YYYY-MM-DD"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run with file logging to test directory
cargo run -- --log-file --log-dir "$LOG_DIR" --no-storage
