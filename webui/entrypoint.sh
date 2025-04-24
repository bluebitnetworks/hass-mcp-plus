#!/bin/sh

# Replace environment variables in the Nginx configuration
API_URL=${API_URL:-http://hass-mcp-plus:8787}

# Update Nginx configuration with actual API URL
sed -i "s|__API_URL__|$API_URL|g" /etc/nginx/conf.d/default.conf

# Update environment.js with runtime environment variables
cat <<EOF > /usr/share/nginx/html/environment.js
window.env = {
  API_URL: "$API_URL",
  VERSION: "${VERSION:-1.0.0}",
  ENV: "${ENV:-production}"
};
EOF

# Execute the CMD
exec "$@"
