### download arm64 version of python via anaconda
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```
## follow steps to activate conda for your shell... then,
```bash
conda create -n llama python=3.11
```

## Nuke everything and reinstall deps
```bash
rm -rf venv build dist
python3 -m venv venv
CMAKE_ARGS="-DLLAMA_METAL=on" ./venv/bin/pip install -e .
```

# Build portable bundle
```bash
./venv/bin/pyinstaller -y rift.spec
```

# Zip it
```bash
export V="2.1.2" && cd dist &&  mv rift rift-macOS-arm64-v$V && zip -r rift-macOS-arm64-v$V.zip rift-macOS-arm64-v$V
```
