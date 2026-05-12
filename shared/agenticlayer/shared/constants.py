"""Constants shared across the agenticlayer package."""

# Prefix used to store propagated HTTP headers in ADK session state as flat primitive keys.
# Each header is stored as a separate string entry: f"{HTTP_HEADERS_SESSION_KEY}.{header_name_lower}"
# e.g. "http_headers.authorization" -> "Bearer token"
HTTP_HEADERS_SESSION_KEY = "http_headers"

# A2A well-known path served by every A2A agent. Vendored here so that `shared`
# does not need to depend on a2a-sdk (the only constant it ever needed).
AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"
