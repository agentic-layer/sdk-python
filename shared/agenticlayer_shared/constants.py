"""Constants shared across the agenticlayer package."""

# Prefix used to store propagated HTTP headers in ADK session state as flat primitive keys.
# Each header is stored as a separate string entry: f"{HTTP_HEADERS_SESSION_KEY}.{header_name_lower}"
# e.g. "http_headers.authorization" -> "Bearer token"
HTTP_HEADERS_SESSION_KEY = "http_headers"
