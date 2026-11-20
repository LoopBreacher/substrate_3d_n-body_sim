#!/usr/bin/env python3
"""Fetch Solar System barycentric Ecliptic J2000.0 state vectors from NASA/JPL Horizons.

Zero third-party dependencies (standard library only).
Outputs CSV directly consumable by:
    substrate_native.exe --epoch-csv <output.csv>

Example:
    python tools/fetch_horizons_epoch.py --date "2029-04-01 00:00" --out epoch_apophis_2029.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

# Simulator heavy-body name -> JPL Horizons COMMAND string (NAIF / SPK / IAU designation)
BODY_COMMANDS: List[Tuple[str, str]] = [
    ("Sun", "10"),
    ("Mercury", "199"),
    ("Venus", "299"),
    ("Earth", "399"),
    ("Moon", "301"),
    ("Mars", "499"),
    ("Phobos", "401"),
    ("Deimos", "402"),
    ("Jupiter", "599"),
    ("Io", "501"),
    ("Europa", "502"),
    ("Ganymede", "503"),
    ("Callisto", "504"),
    ("Saturn", "699"),
    ("Titan", "606"),
    ("Mimas", "601"),
    ("Enceladus", "602"),
    ("Tethys", "603"),
    ("Dione", "604"),
    ("Rhea", "605"),
    ("Iapetus", "608"),
    ("Uranus", "799"),
    ("Miranda", "705"),
    ("Ariel", "701"),
    ("Umbriel", "702"),
    ("Titania", "703"),
    ("Oberon", "704"),
    ("Neptune", "899"),
    ("Triton", "801"),
    ("Proteus", "808"),
    ("Nereid", "802"),
    ("Ceres", "1;"),
    ("Pluto", "999"),
    ("Charon", "901"),
    ("Haumea", "136108;"),
    ("Makemake", "136472;"),
    ("Eris", "136199;"),
    ("Sedna", "90377;"),
    ("Vesta", "4;"),
    ("Pallas", "2;"),
    ("Hygiea", "10;"),
    ("588 Achilles (Jup L4)", "588;"),
    ("617 Patroclus (Jup L5)", "617;"),
    ("2010 TK7 (Earth L4)", "DES=2010 TK7;"),
    ("1P/Halley", "DES=1P; CAP"),
    ("2060 Chiron", "2060;"),
    ("10199 Chariklo", "10199;"),
    ("5145 Pholus", "5145;"),
    ("50000 Quaoar", "50000;"),
    ("225088 Gonggong", "225088;"),
    ("90482 Orcus", "90482;"),
    ("99942 Apophis", "99942;"),
    ("101955 Bennu", "101955;"),
    ("67P/Churyumov-Gerasimenko", "DES=67P; CAP"),
    ("1I/'Oumuamua", "DES=A/2017 U1;"),
]


def parse_epoch_date(date_str: str) -> Tuple[str, str]:
    """Parse 'YYYY-MM-DD[ HH:MM[:SS]]' and return (start_time, stop_time) 1 day apart."""
    s = date_str.strip()
    dt: Optional[datetime.datetime] = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            break
        except ValueError:
            continue
    if dt is None:
        raise ValueError(
            f"Unsupported date format '{date_str}'. Use 'YYYY-MM-DD HH:MM' (e.g. '2000-01-01 12:00')."
        )
    stop_dt = dt + datetime.timedelta(days=1)
    return dt.strftime("%Y-%m-%d %H:%M:%S"), stop_dt.strftime("%Y-%m-%d %H:%M:%S")


def build_ssl_context(ca_bundle: Optional[str], insecure: bool) -> ssl.SSLContext:
    """Build a TLS context supporting Windows system CAs, custom bundles, or opt-in insecure mode."""
    if insecure:
        return ssl._create_unverified_context()
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)

    # Prefer native OS trust store (Windows Schannel / macOS Security) if truststore is installed
    try:
        import truststore  # type: ignore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        pass

    ctx = ssl.create_default_context()

    # Explicitly load Windows ROOT and CA certificate stores (fixes antivirus/proxy self-signed CAs
    # in Conda/MSYS2/standalone Python builds without any third-party packages)
    if sys.platform == "win32" and hasattr(ssl, "enum_certificates"):
        for store_name in ("ROOT", "CA"):
            try:
                for cert_bytes, encoding, trust in ssl.enum_certificates(store_name):
                    if encoding != "x509_asn":
                        continue
                    if trust is True or (isinstance(trust, set) and "1.3.6.1.5.5.7.3.1" in trust):
                        try:
                            ctx.load_verify_locations(cadata=cert_bytes)
                        except ssl.SSLError:
                            continue
            except OSError:
                continue

    # Also load certifi's CA bundle if available
    try:
        import certifi  # type: ignore
        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        pass

    return ctx


def query_horizons_vector(
    command: str,
    start_time: str,
    stop_time: str,
    ssl_context: ssl.SSLContext,
    timeout: float = 20.0,
) -> Tuple[float, ...]:
    """Query NASA/JPL Horizons for a single barycentric ecliptic J2000.0 state vector (AU, AU/day)."""
    params: Dict[str, str] = {
        "format": "json",
        "COMMAND": f"'{command}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@0'",
        "START_TIME": f"'{start_time}'",
        "STOP_TIME": f"'{stop_time}'",
        "STEP_SIZE": "'2d'",
        "REF_PLANE": "'ECLIPTIC'",
        "REF_SYSTEM": "'ICRF'",
        "OUT_UNITS": "'AU-D'",
        "VEC_TABLE": "'2'",
        "CSV_FORMAT": "'YES'",
    }
    url = f"{HORIZONS_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "SubstrateLogisticsSim/1.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_context) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    result_text = payload.get("result", "")
    if "$$SOE" not in result_text or "$$EOE" not in result_text:
        err = payload.get("error") or result_text[:240].replace("\n", " ")
        raise RuntimeError(f"Horizons returned no ephemeris block: {err}")

    block = result_text.split("$$SOE", 1)[1].split("$$EOE", 1)[0].strip()
    first_line = next((ln.strip() for ln in block.splitlines() if ln.strip()), "")
    fields = [c.strip() for c in first_line.split(",")]
    # Horizons CSV VEC_TABLE=2 layout: JDTDB, Calendar Date, X, Y, Z, VX, VY, VZ, ...
    if len(fields) < 8:
        raise RuntimeError(f"Unexpected Horizons CSV row: {first_line}")
    return tuple(float(fields[k]) for k in range(2, 8))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download NASA/JPL Horizons Ecliptic J2000.0 barycentric states for Substrate Logistics."
    )
    parser.add_argument(
        "--date",
        default="2000-01-01 12:00",
        help="TDB epoch date in 'YYYY-MM-DD HH:MM' format (default: '2000-01-01 12:00').",
    )
    parser.add_argument(
        "--out",
        default="epoch_j2000.csv",
        help="Output CSV path (default: epoch_j2000.csv).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Polite delay in seconds between Horizons API requests (default: 0.15).",
    )
    parser.add_argument(
        "--ca-bundle",
        default=None,
        help="Optional path to a custom PEM CA certificate bundle.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (use only behind HTTPS-inspecting proxies/antivirus).",
    )
    args = parser.parse_args()

    start_time, stop_time = parse_epoch_date(args.date)
    ssl_ctx = build_ssl_context(args.ca_bundle, args.insecure)
    print(f"Fetching {len(BODY_COMMANDS)} bodies from NASA/JPL Horizons at TDB epoch {start_time}...")

    rows: List[Tuple[str, float, float, float, float, float, float]] = []
    failed: List[Tuple[str, str]] = []

    for idx, (name, cmd) in enumerate(BODY_COMMANDS, start=1):
        try:
            x, y, z, vx, vy, vz = query_horizons_vector(cmd, start_time, stop_time, ssl_ctx)
            rows.append((name, x, y, z, vx, vy, vz))
            print(f"  [{idx:2d}/{len(BODY_COMMANDS)}] {name:<28s} OK  r=({x:+.6f}, {y:+.6f}, {z:+.6f}) AU")
        except Exception as exc:
            failed.append((name, str(exc)))
            print(f"  [{idx:2d}/{len(BODY_COMMANDS)}] {name:<28s} SKIP ({exc})", file=sys.stderr)
        if args.delay > 0.0 and idx < len(BODY_COMMANDS):
            time.sleep(args.delay)

    if not rows and failed and any("CERTIFICATE_VERIFY_FAILED" in err for _, err in failed):
        print(
            "\nHint: TLS certificate verification failed (common behind HTTPS-inspecting proxies/antivirus). "
            "Retry with --ca-bundle <path/to/ca.pem> or --insecure.",
            file=sys.stderr,
        )

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        f.write(f"# NASA/JPL Horizons Ecliptic J2000.0 Barycentric State Vectors (TDB Epoch: {start_time})\n")
        f.write("# Units: position in AU, velocity in AU/day (CENTER=500@0, REF_PLANE=ECLIPTIC, REF_SYSTEM=ICRF)\n")
        writer = csv.writer(f)
        writer.writerow(["name", "x_au", "y_au", "z_au", "vx_aud", "vy_aud", "vz_aud"])
        for row in rows:
            writer.writerow([
                row[0],
                f"{row[1]:.15e}",
                f"{row[2]:.15e}",
                f"{row[3]:.15e}",
                f"{row[4]:.15e}",
                f"{row[5]:.15e}",
                f"{row[6]:.15e}",
            ])

    print(f"\nWrote {len(rows)} body states to {args.out} ({len(failed)} skipped).")
    print(f"Run with: substrate_native.exe --epoch-csv {args.out}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
