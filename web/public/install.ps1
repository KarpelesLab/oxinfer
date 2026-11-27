# oxinfer installer for Windows:
#
#   irm https://karpeleslab.github.io/oxinfer/install.ps1 | iex
#
# Downloads the latest release's oxinfer.exe from GitHub into %LOCALAPPDATA%\oxinfer\bin (or
# $env:OXINFER_INSTALL_DIR) and adds that folder to the user's PATH. $env:OXINFER_VERSION picks a
# release (e.g. v0.0.12) instead of the latest. A release is published before its binaries are
# attached: when the latest has none for this platform yet, the newest release that has one is
# installed.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repo = 'KarpelesLab/oxinfer'
$dir = if ($env:OXINFER_INSTALL_DIR) { $env:OXINFER_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'oxinfer\bin' }
$tag = $env:OXINFER_VERSION

$arch = $env:PROCESSOR_ARCHITECTURE
if ($env:PROCESSOR_ARCHITEW6432) { $arch = $env:PROCESSOR_ARCHITEW6432 }
$target = switch ($arch) {
    'AMD64' { 'windows-x86_64' }
    'ARM64' { 'windows-aarch64' }
    default { throw "oxinfer installer: no release binary for Windows on $arch" }
}

if (-not $tag) {
    # The latest release, from the redirect of /releases/latest (no API rate limit).
    $req = [Net.WebRequest]::Create("https://github.com/$repo/releases/latest")
    $req.AllowAutoRedirect = $false
    $resp = $req.GetResponse()
    $loc = $resp.Headers['Location']
    $resp.Close()
    $tag = ($loc -split '/tag/')[-1]
    if ($tag -notlike 'v*') { throw "oxinfer installer: could not find the latest release (got $loc)" }
    $has = $true
    try {
        Invoke-WebRequest -UseBasicParsing -Method Head -Uri "https://github.com/$repo/releases/download/$tag/oxinfer-$tag-$target.zip" | Out-Null
    } catch { $has = $false }
    if (-not $has) {
        # Its binary for this platform is not attached (yet): the newest release that has one,
        # from the API's list of releases (newest first).
        Write-Host "oxinfer $tag has no $target binary yet; looking at the releases before it"
        $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases?per_page=15" -Headers @{ 'User-Agent' = 'oxinfer-installer' }
        $prev = $releases | Where-Object {
            $t = $_.tag_name
            -not $_.draft -and -not $_.prerelease -and ($_.assets | Where-Object { $_.name -eq "oxinfer-$t-$target.zip" })
        } | Select-Object -First 1
        if (-not $prev) { throw "oxinfer installer: no release has a $target binary" }
        $tag = $prev.tag_name
    }
}

$archive = "oxinfer-$tag-$target.zip"
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("oxinfer-" + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host "Downloading oxinfer $tag ($target)"
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/$repo/releases/download/$tag/$archive" -OutFile (Join-Path $tmp $archive)
    Expand-Archive -Path (Join-Path $tmp $archive) -DestinationPath $tmp -Force
    $exe = Join-Path $tmp 'oxinfer.exe'
    if (-not (Test-Path $exe)) { throw 'oxinfer installer: the archive has no oxinfer.exe' }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $dest = Join-Path $dir 'oxinfer.exe'
    # A running oxinfer.exe cannot be overwritten but can be renamed out of the way.
    if (Test-Path $dest) {
        $old = "$dest.old"
        Remove-Item $old -Force -ErrorAction SilentlyContinue
        Move-Item $dest $old -Force
    }
    Copy-Item $exe $dest
    Write-Host "Installed $dest"
} finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not (($userPath -split ';') -contains $dir)) {
    [Environment]::SetEnvironmentVariable('Path', ($(if ($userPath) { "$userPath;" } else { '' }) + $dir), 'User')
    $env:Path = "$env:Path;$dir"
    Write-Host "Added $dir to your PATH (open a new terminal for other windows to see it)."
}

Write-Host ''
Write-Host 'Get started:  oxinfer web     (the GUI in your browser)'
Write-Host '              oxinfer --help'
Write-Host 'Update later: oxinfer --update'
