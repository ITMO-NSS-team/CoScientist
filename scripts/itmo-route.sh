#!/bin/bash
# Pin a route to the ITMO 10.32.0.0/16 network through the FortiClient tunnel.
#
# Why this exists: when both the ITMO FortiClient VPN and another full-tunnel VPN
# (e.g. "Happ Plus") are connected, Happ Plus's exclude-route for 10.0.0.0/8 sends
# ITMO traffic out the physical interface, shadowing FortiClient's own 10/8 route.
# A manual `route add ... -interface utunN` fixes it, but FortiClient recreates its
# utun (and its 10.64.x address) on every reconnect, dropping that route. This
# script re-pins the route via whatever utun currently holds the Forti-assigned
# 10.x address. Safe to run repeatedly; no-ops when Forti is down or already pinned.
set -u

# FortiClient tunnel = the first utun carrying a 10.x address (Happ Plus is 198.18.x).
IFACE=$(ifconfig | awk '/^utun[0-9]+:/{n=$1; sub(":","",n)} /inet 10\./{print n; exit}')
[ -z "${IFACE:-}" ] && exit 0   # FortiClient not connected -> nothing to do

cur=$(route -n get 10.32.11.45 2>/dev/null | awk '/interface:/{print $2}')
if [ "$cur" != "$IFACE" ]; then
  /sbin/route -n delete -net 10.32.0.0/16 >/dev/null 2>&1
  /sbin/route -n add -net 10.32.0.0/16 -interface "$IFACE" >/dev/null 2>&1 \
    && logger "itmo-route: pinned 10.32.0.0/16 -> $IFACE"
fi
