# -*- mode: python ; coding: utf-8 -*-


from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
import os



a = Analysis(
    ['gui_v8.py'],
    pathex=['D:\\Workspace Desktop\\GAN'],
    binaries=[],
    datas=[
        ('ganGenerator.onnx', '.'),          # ONNX-Modell
        ('generator.py', '.'),               # Falls du ihn separat importierst
        ('fonts/Old_Standard_TT.ttf', 'fonts'),             # Fonts, Bilder etc.
        ('assets/frame0/button_1.png', 'assets/frame0'),
        ('assets/frame0/dummy.jpg', 'assets/frame0'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='gui_v8',
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
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='gui_v8',
)
