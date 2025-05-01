#!/bin/bash

# Build the frontend using the Vite build system
echo "Building FasterLivePortrait Three.js viewer..."

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

# Build the project
echo "Building project..."
npm run build

# Create a symbolic link to the dist directory in the project root for convenience
echo "Creating symbolic link to dist directory..."
if [ ! -d "../static" ]; then
  mkdir -p ../static
fi

# Link all files from dist to static
ln -sf $(pwd)/dist/* ../static/

echo "Build complete. Frontend files available in frontend/dist/ and linked to static/"