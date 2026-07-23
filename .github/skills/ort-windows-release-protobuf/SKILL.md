---
name: ort-windows-release-protobuf
description: Fixes a Windows-only ONNX Runtime Release build failure where the link step reports LNK2038 _ITERATOR_DEBUG_LEVEL / RuntimeLibrary mismatches against libprotobuf-lited.lib. Use when building ORT on Windows (VS 2022 generator) and the link fails with a Debug protobuf being pulled into the Release DLL, or when configuring a fresh build_dir on a box that has a stray external protobuf installed. Complements the ort-build skill.
---

# Building ONNX Runtime Release on Windows: force the bundled protobuf

## Symptom

A Release CPU (or any Release) build configures fine, compiles for a long time, then
**fails at link** with many lines like:

```
libprotobuf-lited.lib(message_lite.obj) : error LNK2038: mismatch detected for
  '_ITERATOR_DEBUG_LEVEL': value '2' doesn't match value '0' in dllmain.obj
  [ ...\build\Windows\Release\onnxruntime.vcxproj]
libprotobuf-lited.lib(...) : error LNK2038: mismatch detected for 'RuntimeLibrary':
  value 'MDd_DynamicDebug' doesn't match value 'MD_DynamicRelease' in dllmain.obj
```

The tell-tale sign is the **`d` suffix** in `libprotobuf-lite**d**.lib` — that is protobuf's
`DEBUG_POSTFIX`, i.e. a **Debug** protobuf being linked into a **Release** `onnxruntime.dll`.

## Root cause

ORT declares protobuf via FetchContent with a `find_package` fallback. If CMake's **user
package registry** (or `CMAKE_PREFIX_PATH`) points at a pre-built external protobuf, FetchContent
uses that instead of building ORT's bundled copy. When that external protobuf was built
**Debug-only** (e.g. a stray `C:\repos\protobuf_build\lib\libprotobuf-lited.lib`, often an older
version like 3.20.2 vs ORT's expected 21.12), it gets linked into the Release DLL and the CRT /
iterator-debug flags collide.

Confirm the culprit by checking the generated cache:

```powershell
Select-String -Path build\Windows\Release\CMakeCache.txt -Pattern "Protobuf_DIR"
# e.g. Protobuf_DIR:PATH=C:/repos/protobuf_build/cmake  <- external, wrong
```

The configure log also prints `Using protobuf from find_package(or vcpkg). Protobuf version:
3.20.2.0` instead of building `_deps/protobuf-src`.

## Fix

Force FetchContent to **never** use find_package, so ORT builds its bundled protobuf (v21.12)
from source. Add to the build command:

```
--cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
```

Full working Release + CPU + wheel command on Windows (VS 2022):

```powershell
python tools\ci_build\build.py --build_dir build\Windows --config Release --parallel `
  --build_wheel --skip_tests --cmake_generator "Visual Studio 17 2022" `
  --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
```

After configure, verify the log shows protobuf building from source
(`_deps/protobuf-src/CMakeLists.txt`, `Fetch Protobuf ... v21.12.zip`) and **not**
`Using protobuf from find_package`.

Notes:
- `FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER` only affects FetchContent-declared deps with a
  `find_package` fallback (protobuf, onnx, abseil, …). It does **not** disable plain
  `find_package` for Python/CUDA, so it is safe and matches CI behavior.
- If you had a stale/failed config in the build dir, the cached `Protobuf_DIR` can persist; the
  NEVER mode bypasses find_package entirely so it no longer matters, but a clean re-configure
  (`--update`) is the surest reset.

## Running the resulting wheel without repo shadowing

After `--build_wheel`, pip-install the wheel into a dedicated venv and run from **any directory
except the repo root**. The repo root contains an `onnxruntime/` source package dir that shadows
the install (`ModuleNotFoundError: No module named 'onnxruntime.capi'`). From a neutral cwd,
`import onnxruntime` correctly resolves to the venv site-packages.
