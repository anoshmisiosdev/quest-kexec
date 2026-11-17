#!/usr/bin/env python3
"""Prepare Image + initramfs + DTB for the Quest Pro kexec staging layout.

The Android boot image is optional; it is only needed when the kernel or
initramfs is taken from it.

Only writes host files. Does not contact the headset or execute a boot.
"""
import argparse
import gzip
import hashlib
import json
import struct
import subprocess
import zlib
from pathlib import Path

# Destination window in physical RAM.  Must match module/loader.c exactly.
# The initramfs window (QKX_DTB - QKX_INITRD) was 2 MiB and is grown to 48 MiB
# so a full OS ramdisk fits (SteamOS: ~19 MiB internal, ~39 MiB USB installer);
# the DTB keeps 2 MiB.  The window stays in one clean System RAM band on Quest
# Pro and Quest 2 (above the top firmware carveout, below the next one up), and
# check_ram() below still re-validates it against the live /proc/iomem.
QKX_LO = 0x90000000
QKX_ENTRY = 0x90080000
QKX_INITRD = 0x93000000
QKX_DTB = 0x96000000
QKX_HI = 0x96200000
QKX_IMAGE_MAX = QKX_INITRD - QKX_ENTRY   # kernel image_size ceiling (47.5 MiB)
QKX_INITRD_MAX = QKX_DTB - QKX_INITRD    # 48 MiB
QKX_DTB_MAX = QKX_HI - QKX_DTB           # 2 MiB


def align(n, a):
    return (n + a - 1) & -a


def unpack(data):
    if len(data) < 1660 or data[:8] != b"ANDROID!":
        raise ValueError("not an Android boot image")
    ksize, _, rsize, _, ssize, _, _, page, version, _ = struct.unpack_from("<10I", data, 8)
    if version not in (0, 1, 2):
        raise ValueError("only boot header versions 0, 1 and 2 are supported")
    if page not in (2048, 4096, 8192, 16384) or not ksize:
        raise ValueError("invalid page or kernel size")
    if ssize:
        raise ValueError("second-stage bootloader payloads are not supported")
    roff = align(page + ksize, page)
    soff = align(roff + rsize, page)
    if soff + ssize > len(data) or roff + rsize > len(data):
        raise ValueError("truncated boot image")
    kernel = data[page:page + ksize]
    if kernel.startswith(b"\x1f\x8b"):
        # Gzip streams with appended DTBs need separate handling, not guesses.
        kernel = gzip.decompress(kernel)
    if len(kernel) < 64 or kernel[56:60] != b"ARM\x64":
        raise ValueError("kernel is not an uncompressed/gzip ARM64 Image")
    text_offset, image_size, flags = struct.unpack_from("<QQQ", kernel, 8)
    if not image_size or image_size < len(kernel) or flags & 1:
        raise ValueError("unsupported ARM64 image size/endianness (appended data must be separated)")
    if flags & 6 not in (0, 2):
        raise ValueError("this loader requires a 4K-page target")
    if not flags & 8:
        raise ValueError("target does not permit placement away from the base of RAM")
    cmdline = (data[64:576].split(b"\0", 1)[0] +
               data[608:1632].split(b"\0", 1)[0]).decode("ascii")
    return kernel, data[roff:roff + rsize], dict(
        boot_header_version=version, text_offset=text_offset,
        image_size=image_size, image_file_size=len(kernel), flags=flags,
        cmdline=cmdline, boot_sha256=hashlib.sha256(data).hexdigest())


def check_ram(iomem, start, size):
    """Require one System RAM extent and reject every nested reserved extent."""
    end = start + size
    ranges = []
    for line in iomem.splitlines():
        span, sep, label = line.strip().partition(" : ")
        if not sep:
            continue
        lo, hi = (int(x, 16) for x in span.split("-"))
        ranges.append((lo, hi + 1, label))
    if not any(lo <= start < end <= hi and label == "System RAM" for lo, hi, label in ranges):
        raise ValueError("destination does not fit in a captured System RAM extent")
    for lo, hi, label in ranges:
        if label != "System RAM" and start < hi and lo < end:
            raise ValueError(f"destination overlaps {label}: {lo:#x}-{hi:#x}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=Path, help="Android boot.img (v0-v2) to take missing parts from")
    ap.add_argument("--live-dtb", type=Path, required=True)
    ap.add_argument("--iomem", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--initramfs", type=Path, help="optional replacement; original is retained separately")
    ap.add_argument("--cmdline", help="explicit replacement command line")
    ap.add_argument("--kernel-image", type=Path, help="replacement raw ARM64 Image")
    ap.add_argument("--keep-watchdog", action="store_true",
                    help="leave /soc/qcom,wdt@17c10000 enabled in the DTB")
    a = ap.parse_args()
    if a.boot:
        kernel, original, report = unpack(a.boot.read_bytes())
    elif a.kernel_image and a.initramfs:
        kernel, original, report = b"", b"", dict(cmdline="")
    else:
        ap.error("--boot is required unless --kernel-image and --initramfs are both given")
    if a.kernel_image:
        kernel = a.kernel_image.read_bytes()
        if len(kernel) < 64 or kernel[56:60] != b"ARM\x64":
            raise ValueError("replacement is not a raw ARM64 Image")
        offset, size, flags = struct.unpack_from("<QQQ", kernel, 8)
        if size < len(kernel) or flags & 1 or flags & 6 not in (0, 2) or not flags & 8:
            raise ValueError("unsupported replacement Image")
        report.update(text_offset=offset, image_size=size, flags=flags,
                      image_file_size=len(kernel), replacement_kernel=str(a.kernel_image),
                      kernel_sha256=hashlib.sha256(kernel).hexdigest())
    if report["text_offset"] != 0x80000 or report["image_size"] > QKX_IMAGE_MAX:
        raise ValueError("image does not fit the kexec staging layout")
    initrd = a.initramfs.read_bytes() if a.initramfs else original
    dtb = a.live_dtb.read_bytes()
    if len(dtb) < 40 or struct.unpack_from(">I", dtb)[0] != 0xD00DFEED:
        raise ValueError("invalid live DTB")
    if len(initrd) > QKX_INITRD_MAX or len(dtb) > QKX_DTB_MAX:
        raise ValueError("initramfs/DTB exceeds staging limits")
    # The whole destination window must be one System RAM extent, free of any
    # reserved sub-region, in the live map.
    check_ram(a.iomem.read_text(), QKX_LO, QKX_HI - QKX_LO)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "Image").write_bytes(kernel)
    if original:
        (a.output / "original-ramdisk.gz").write_bytes(original)
    (a.output / "initramfs").write_bytes(initrd)
    dtpath = a.output / "boot.dtb"
    dtpath.write_bytes(dtb)
    args = a.cmdline if a.cmdline is not None else report["cmdline"]
    subprocess.run(["fdtput", "-t", "s", str(dtpath), "/chosen", "bootargs", args], check=True)
    # The target kernel takes over the apps watchdog from the loader instead of
    # probing it again; this matches every validated boot.
    if not a.keep_watchdog:
        subprocess.run(["fdtput", "-t", "s", str(dtpath), "/soc/qcom,wdt@17c10000",
                        "status", "disabled"], check=True)
    for name, address in (("linux,initrd-start", QKX_INITRD),
                          ("linux,initrd-end", QKX_INITRD + len(initrd))):
        subprocess.run(["fdtput", "-t", "x", str(dtpath), "/chosen", name,
                        "0", f"{address:x}"], check=True)
    report.update(kernel_address=QKX_ENTRY, initrd_address=QKX_INITRD,
                  dtb_address=QKX_DTB, initrd_size=len(initrd),
                  dtb_size=dtpath.stat().st_size, effective_cmdline=args,
                  replacement_initramfs=a.initramfs is not None,
                  crc32={name: f"{zlib.crc32(content):08x}" for name, content in
                         (("Image", kernel), ("initramfs", initrd), ("boot.dtb", dtpath.read_bytes()))},
                  status="prepared only; no boot attempted")
    (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
