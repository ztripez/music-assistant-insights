# Multi-stage Dockerfile for insight-sidecar (CPU version)
# Supports multi-arch builds (amd64, arm64)

ARG RUST_VERSION=1.83
ARG BASE_IMAGE=debian:bookworm-slim
ARG FEATURES=storage,storage-file,audio,watcher

# =============================================================================
# Builder stage
# =============================================================================
FROM --platform=$BUILDPLATFORM rust:${RUST_VERSION}-bookworm AS builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG FEATURES

# Install cross-compilation tools for ARM64 if needed
RUN case "$TARGETPLATFORM" in \
    "linux/arm64") \
        apt-get update && apt-get install -y \
            gcc-aarch64-linux-gnu \
            g++-aarch64-linux-gnu \
            libc6-dev-arm64-cross \
        && rm -rf /var/lib/apt/lists/* \
        && rustup target add aarch64-unknown-linux-gnu \
        ;; \
    esac

WORKDIR /build

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./

# Create dummy source to cache dependencies
RUN mkdir -p src && \
    echo 'fn main() { println!("dummy"); }' > src/main.rs

# Pre-build dependencies
RUN case "$TARGETPLATFORM" in \
    "linux/arm64") \
        export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc && \
        cargo build --release --target aarch64-unknown-linux-gnu --features "$FEATURES" || true \
        ;; \
    *) \
        cargo build --release --features "$FEATURES" || true \
        ;; \
    esac

# Copy actual source code
COPY src ./src

# Build the actual binary
RUN case "$TARGETPLATFORM" in \
    "linux/arm64") \
        export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc && \
        cargo build --release --target aarch64-unknown-linux-gnu --features "$FEATURES" && \
        aarch64-linux-gnu-strip /build/target/aarch64-unknown-linux-gnu/release/insight-sidecar && \
        cp /build/target/aarch64-unknown-linux-gnu/release/insight-sidecar /build/insight-sidecar \
        ;; \
    *) \
        cargo build --release --features "$FEATURES" && \
        strip /build/target/release/insight-sidecar && \
        cp /build/target/release/insight-sidecar /build/insight-sidecar \
        ;; \
    esac

# =============================================================================
# Runtime stage
# =============================================================================
FROM ${BASE_IMAGE}

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r sidecar && useradd -r -g sidecar sidecar

# Copy binary from builder
COPY --from=builder /build/insight-sidecar /usr/local/bin/insight-sidecar

# Create data directory
RUN mkdir -p /data && chown -R sidecar:sidecar /data

# Switch to non-root user
USER sidecar

# Set default environment variables
ENV INSIGHT_SERVER__HOST=0.0.0.0
ENV INSIGHT_SERVER__PORT=8096
ENV INSIGHT_STORAGE__MODE=file
ENV INSIGHT_STORAGE__DATA_DIR=/data

# Expose the default port
EXPOSE 8096

# Set volume for persistent data
VOLUME ["/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8096/health || exit 1

# Run the sidecar
ENTRYPOINT ["insight-sidecar"]
