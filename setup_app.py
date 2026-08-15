"""py2app build script for the TRGB .app wrapper.

Build with:

    python3 setup_app.py py2app -A

Alias mode (-A) makes the bundle run this repo's code in place instead of
freezing a copy: the GUI reads and writes galaxy data relative to the repo,
and code edits take effect on next launch without rebuilding. The bundle is
therefore machine-local -- rebuild it after moving the repo, don't ship it.
"""

from setuptools import setup

setup(
    name="TRGB",
    app=["trgb_gui.py"],
    options={"py2app": {
        "iconfile": "gui/icon.icns",
        "plist": {
            "CFBundleName": "TRGB",
            "CFBundleDisplayName": "TRGB",
            "CFBundleIdentifier": "edu.cornell.surf.trgb",
            "NSHighResolutionCapable": True,
        },
    }},
)
