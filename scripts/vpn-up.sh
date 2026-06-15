#!/bin/bash
# Pre-test VPN healer: ensure the ITMO FortiClient tunnel is connected, then verify
# ITMO 10.32.x is reachable. The 10.32 route itself is kept pinned by the
# `local.itmo.route` launchd daemon (one-time install — see scripts/README or the
# commands below), so this script needs NO sudo.
#
# One-time route auto-fixer install (run once, at the keyboard):
#   sudo cp scripts/itmo-route.sh /usr/local/sbin/itmo-route.sh
#   sudo chown root:wheel /usr/local/sbin/itmo-route.sh && sudo chmod 755 /usr/local/sbin/itmo-route.sh
#   sudo cp scripts/local.itmo.route.plist /Library/LaunchDaemons/local.itmo.route.plist
#   sudo chown root:wheel /Library/LaunchDaemons/local.itmo.route.plist
#   sudo launchctl load -w /Library/LaunchDaemons/local.itmo.route.plist
#
# Exit 0 = ITMO reachable; exit 2 = not reachable (Forti couldn't connect or route unpinned).
set -u
FORTI_ID="85300411-3CC9-4C4D-BBAD-1F3D8582CDA8"   # com.fortinet.forticlient.macos.vpn

python3 - "$FORTI_ID" <<'PY'
import subprocess, sys, time, socket
fid = sys.argv[1]

def status():
    out = subprocess.run(["scutil", "--nc", "status", fid], capture_output=True, text=True).stdout
    return (out.splitlines() or [""])[0].strip()

if status() != "Connected":
    print(f"[vpn-up] FortiClient '{status()}' -> starting (scutil --nc start)")
    subprocess.run(["scutil", "--nc", "start", fid])
    for _ in range(25):                 # wait up to ~50s for the tunnel
        if status() == "Connected":
            break
        time.sleep(2)
print(f"[vpn-up] FortiClient: {status()}")

def up(host, port, t=5):
    try:
        c = socket.create_connection((host, port), t); c.close(); return True
    except Exception:
        return False

ok = up("10.32.2.2", 8764) and up("10.32.1.36", 6333)
print(f"[vpn-up] ITMO 10.32.x reachable: {ok}")
if not ok and status() == "Connected":
    print("[vpn-up] NOTE: Forti up but ITMO unreachable -> the 10.32 route is not pinned. "
          "Install the local.itmo.route launchd daemon (see header), or run "
          "`sudo /usr/local/sbin/itmo-route.sh`.")
sys.exit(0 if ok else 2)
PY
