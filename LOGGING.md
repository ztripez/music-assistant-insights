# File Logging

The sidecar supports optional file logging for debugging and session sharing.

## Usage

### Enable file logging with default location

```bash
./insight-sidecar --log-file
```

Logs will be written to: `~/.local/share/insight-sidecar/logs/insight-sidecar.log.YYYY-MM-DD`

### Enable file logging with custom directory

```bash
./insight-sidecar --log-file --log-dir /path/to/logs
```

### Using environment variables

```bash
export INSIGHT_LOG_FILE=true
export INSIGHT_LOG_DIR=/path/to/logs  # optional
./insight-sidecar
```

## Log Rotation

Logs are automatically rotated daily. Each day creates a new file with the format:
- `insight-sidecar.log.2025-12-26`
- `insight-sidecar.log.2025-12-27`
- etc.

## Log Format

When logging to file:
- Console output: includes ANSI colors for readability
- File output: plain text without colors for easy parsing

Both outputs include:
- Timestamp
- Log level (INFO, WARN, ERROR, DEBUG, TRACE)
- Module name
- Message
- Structured fields (when available)

## Verbosity Levels

Combine with verbosity flags:

```bash
# Default (INFO level)
./insight-sidecar --log-file

# Debug level
./insight-sidecar --log-file -v

# Trace level (very verbose)
./insight-sidecar --log-file -vv

# Quiet mode (WARN and ERROR only)
./insight-sidecar --log-file --quiet
```

## Sharing Session Logs

To share a session run with someone:

1. Start sidecar with file logging:
   ```bash
   ./insight-sidecar --log-file -v
   ```

2. Reproduce the issue or run your test

3. Find the log file:
   ```bash
   ls -lt ~/.local/share/insight-sidecar/logs/
   ```

4. Share the most recent log file

## Example Session

```bash
# Start with file logging and debug verbosity
./insight-sidecar --log-file -v

# In another terminal, make API calls
curl http://localhost:8096/api/v1/health

# Stop sidecar (Ctrl+C)

# View the log
cat ~/.local/share/insight-sidecar/logs/insight-sidecar.log.$(date +%Y-%m-%d)
```

## Log Contents

The logs capture:
- Server startup and configuration
- Model loading and initialization
- Storage connection and setup
- All API requests and responses
- Streaming session lifecycle
- Error details with stack traces
- Performance metrics
- Graceful shutdown

This makes it easy to diagnose issues or share session details for troubleshooting.
