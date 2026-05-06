# Publishing Quadtrix.cpp

This repository can be published in two useful ways:

- npm package: gives users a `quadtrix` command for chat and local training.
- GitHub Release: publishes a bundled archive from a version tag.

For most users, npm is the friendliest install path.

## npm Package

The root `package.json` publishes a CLI package named `quadtrix`.

Users will be able to run:

```bash
npm install -g quadtrix
quadtrix setup
quadtrix chat
quadtrix train --backend cpp --data ./data/input.txt
quadtrix train --backend python --data ./data/input.txt
```

`quadtrix setup` installs the Python dependencies needed by the backend and
PyTorch training path. `quadtrix chat` starts the FastAPI backend, serves the
built frontend, and opens the browser. `quadtrix train` lets the user choose the
C++ or Python training backend. Training expects the user to pass a local text
dataset with `--data`.

Before publishing to npm:

```bash
npm login
npm pack --dry-run
npm publish
```

If the package name `quadtrix` is already taken on npm, change the root
`package.json` name to a scoped package such as:

```json
"name": "@eamon2009/quadtrix"
```

Then publish with:

```bash
npm publish --access public
```

## Before Publishing

1. Make sure the working tree only contains files you want to publish:

   ```bash
   git status
   ```

2. Do not commit build outputs, virtual environments, model checkpoints, or local
   binaries. The current `.gitignore` already excludes common examples such as
   `.venv`, `build`, `frontend/dist`, `*.exe`, and `*best_model.pt`.

3. Run the project checks locally when possible:

   ```bash
   g++ -std=c++17 -O2 -I. -Iinclude -o quadtrix main.cpp
   python -m compileall backend engine iGPU
   cd frontend
   npm ci
   npm run build
   ```

## Publish a Release

GitHub will publish a release automatically when you push a version tag that
starts with `v`.

```bash
git add .
git commit -m "Prepare release"
git push origin master

git tag v1.0.0
git push origin v1.0.0
```

After the tag is pushed, open the repository on GitHub and go to:

```text
Actions -> Release
```

When the workflow finishes, the new release will be available under:

```text
Releases
```

The release contains `quadtrix-linux.tar.gz`, including the Linux C++ binary,
Python backend/engine files, frontend build output, README, license, and run
guide.

## Publish the Repository Itself

This repository already has a GitHub remote configured:

```text
https://github.com/Eamon2009/Quadtrix-2.0.git
```

To publish normal code changes:

```bash
git add .
git commit -m "Describe your change"
git push origin master
```
