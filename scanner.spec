# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['scanner.py'],
             pathex=['E:\\Proiecte_Andrei\\Facultate\\Licenta\\RecursiveCNN\\RecursiveCNN\\Recursive-CNNs-server_branch'],
             binaries=[],
             datas=[],
             hiddenimports=['FileDialog'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=True)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [('v', None, 'OPTION')],
          exclude_binaries=True,
          name='scanner',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               upx_exclude=[],
               name='scanner')
