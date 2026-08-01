[app]
title = GeminiWatermarkEraser
project_dir = .
input_file = app.py
exec_directory = dist
project_file =
icon =

[python]
python_path =
packages = Nuitka==2.8.10
android_packages =

[qt]
qml_files =
excluded_qml_plugins = QtQuick,QtQuick3D,QtCharts,QtWebEngine,QtTest,QtSensors
modules = Core,Gui,Widgets
plugins = platforms,imageformats,iconengines,styles

[android]
wheel_pyside =
wheel_shiboken =
plugins =

[nuitka]
macos.permissions =
mode = standalone
extra_args = --quiet --noinclude-qt-translations --windows-console-mode=disable --output-filename=GeminiWatermarkEraser.exe --include-package=gemini_watermark_eraser --include-package-data=imageio_ffmpeg --mingw64 --assume-yes-for-downloads

[buildozer]
mode = debug
recipe_dir =
jars_dir =
ndk_path =
sdk_path =
local_libs =
arch =
