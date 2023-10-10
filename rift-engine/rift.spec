# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import get_package_paths, copy_metadata

block_cipher = None

datas = [
         ('vendor/tree-sitter-rescript/src', 'vendor/tree-sitter-rescript/src' ),
         (get_package_paths('mentat')[1] + '/default_config.json', 'mentat/'),
         (get_package_paths('gpt_engineer')[1] + '/preprompts', 'gpt_engineer/preprompts'),
         (get_package_paths('langchain')[1] + '/chains/llm_summarization_checker/prompts', 'langchain//chains/llm_summarization_checker/prompts'),
         (get_package_paths('gpt4all')[1] + '/llmodel_DO_NOT_MODIFY/build', 'gpt4all/llmodel_DO_NOT_MODIFY/build'),
         (get_package_paths('tree_sitter_languages')[1], 'tree_sitter_languages'),
         (get_package_paths('llama_cpp')[1], 'llama_cpp'),
          # hack to touch "__init__.pyc" as Language#build_library() wants it to exist to check its mtime
         (get_package_paths('tree_sitter')[1] + '/__init__.py', 'tree_sitter/__init__.pyc')
         ]

datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('tokenizers')
datas += copy_metadata('huggingface-hub')
datas += copy_metadata('safetensors')
datas += copy_metadata('transformers')
datas += copy_metadata('pyyaml')

a = Analysis(
    ['rift/server/core.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['tiktoken_ext.openai_public', 'tiktoken_ext', 'tree-sitter-rescript', 'tqdm', 'transformers'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='core',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='rift',
)
