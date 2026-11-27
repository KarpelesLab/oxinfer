#!/bin/sh
# oxinfer installer for macOS and Linux:
#
#   curl -fsSL https://karpeleslab.github.io/oxinfer/install.sh | sh
#
# Downloads the latest release's binary from GitHub into ~/.local/bin (or $OXINFER_INSTALL_DIR).
# A release is published before its binaries are attached: when the latest has none for this
# platform yet, the newest release that has one is installed.
# Options, after `sh -s --`:
#   --dir DIR        install into DIR
#   --version TAG    install release TAG (e.g. v0.0.12) instead of the latest
#   --static         on Linux x86_64, take the static build (no glibc needed)
#   --web            start `oxinfer web` once installed
set -eu

REPO="KarpelesLab/oxinfer"
DIR="${OXINFER_INSTALL_DIR:-$HOME/.local/bin}"
TAG="${OXINFER_VERSION:-}"
STATIC=0
WEB=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) DIR="$2"; shift 2 ;;
        --dir=*) DIR="${1#*=}"; shift ;;
        --version) TAG="$2"; shift 2 ;;
        --version=*) TAG="${1#*=}"; shift ;;
        --static) STATIC=1; shift ;;
        --web) WEB=1; shift ;;
        -h|--help) echo "usage: install.sh [--dir DIR] [--version TAG] [--static] [--web]"; exit 0 ;;
        *) echo "oxinfer installer: unknown option $1" >&2; exit 2 ;;
    esac
done

say() { printf '%s\n' "$*" >&2; }
die() { say "oxinfer installer: $*"; exit 1; }

if command -v curl >/dev/null 2>&1; then
    fetch() { curl -fL --proto '=https' --tlsv1.2 --progress-bar -o "$2" "$1"; }
    exists() { curl -fsSLI --proto '=https' -o /dev/null "$1" 2>/dev/null; }
    text() { curl -fsSL --proto '=https' "$1"; }
    final_url() { curl -fsSLI --proto '=https' -o /dev/null -w '%{url_effective}' "$1"; }
elif command -v wget >/dev/null 2>&1; then
    fetch() { wget -q --show-progress -O "$2" "$1"; }
    exists() { wget -q --spider "$1" 2>/dev/null; }
    text() { wget -q -O - "$1"; }
    final_url() { wget -q --spider -S "$1" 2>&1 | sed -n 's/^ *[Ll]ocation: *//p' | tail -n 1 | tr -d '\r'; }
else
    die "needs curl or wget"
fi

os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
    Darwin) target="macos-universal" ;;
    Linux)
        case "$arch" in
            x86_64|amd64)
                # Both load the GPU libraries at run time where a driver is installed. The static
                # one runs anywhere, without glibc (musl distributions): it needs only a kernel.
                if [ "$STATIC" = 1 ] || ! (ldd --version 2>&1 | grep -qi -e glibc -e 'gnu libc'); then
                    target="linux-x86_64-static"
                else
                    target="linux-x86_64"
                fi ;;
            aarch64|arm64) target="linux-aarch64" ;;
            *) die "no release binary for Linux on $arch; build from source: cargo install oxinfer" ;;
        esac ;;
    MINGW*|MSYS*|CYGWIN*)
        die "on Windows, run in PowerShell: irm https://karpeleslab.github.io/oxinfer/install.ps1 | iex" ;;
    *) die "no release binary for $os; build from source: cargo install oxinfer" ;;
esac

if [ -z "$TAG" ]; then
    # The latest release, from the redirect of /releases/latest (no API rate limit).
    url="$(final_url "https://github.com/$REPO/releases/latest")"
    TAG="${url##*/tag/}"
    case "$TAG" in v*) ;; *) die "could not find the latest release (got $url)" ;; esac
    if ! exists "https://github.com/$REPO/releases/download/$TAG/oxinfer-$TAG-$target.tar.gz"; then
        # Its binary for this platform is not attached (yet): the newest release that has one,
        # from the API's list of releases (newest first).
        say "oxinfer $TAG has no $target binary yet; looking at the releases before it"
        found="$(text "https://api.github.com/repos/$REPO/releases?per_page=15" \
            | grep -o "/releases/download/v[^/\"]*/oxinfer-v[^/\"]*-$target\.tar\.gz" | head -n 1)" || found=""
        prev="${found#/releases/download/}"
        prev="${prev%%/*}"
        case "$prev" in v*) TAG="$prev" ;; *) die "no release has a $target binary; build from source: cargo install oxinfer" ;; esac
    fi
fi

archive="oxinfer-$TAG-$target.tar.gz"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM
say "Downloading oxinfer $TAG ($target)"
fetch "https://github.com/$REPO/releases/download/$TAG/$archive" "$tmp/$archive" \
    || die "download failed: https://github.com/$REPO/releases/download/$TAG/$archive"
tar -xzf "$tmp/$archive" -C "$tmp"
[ -f "$tmp/oxinfer" ] || die "the archive has no oxinfer binary"

mkdir -p "$DIR"
# Replace through a rename, so a running oxinfer keeps its file.
cp "$tmp/oxinfer" "$DIR/.oxinfer.new"
chmod 755 "$DIR/.oxinfer.new"
mv -f "$DIR/.oxinfer.new" "$DIR/oxinfer"
say "Installed $DIR/oxinfer"

case ":$PATH:" in
    *":$DIR:"*) ;;
    *)
        say ""
        say "$DIR is not in your PATH. Add it, e.g.:"
        case "${SHELL:-}" in
            */zsh) say "  echo 'export PATH=\"$DIR:\$PATH\"' >> ~/.zshrc && exec zsh" ;;
            */fish) say "  fish_add_path $DIR" ;;
            *) say "  echo 'export PATH=\"$DIR:\$PATH\"' >> ~/.bashrc && exec bash" ;;
        esac ;;
esac

say ""
say "Get started:  oxinfer web     (the GUI in your browser)"
say "              oxinfer --help"
say "Update later: oxinfer --update"

if [ "$WEB" = 1 ]; then
    exec "$DIR/oxinfer" web
fi
