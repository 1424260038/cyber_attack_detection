"""Shared attack class taxonomy and label normalization."""

from __future__ import annotations


CLASS_NAMES = [
    "Normal",
    "DoS/DDoS",
    "Probe/Scan",
    "R2L",
    "U2R",
    "Malware/Ransomware",
    "Phishing",
    "APT",
]

CANONICAL_ATTACK_TYPES = [
    "normal",
    "dos",
    "probe",
    "r2l",
    "u2r",
    "malware",
    "phishing",
    "apt",
]

LABEL_ALIASES = {
    "normal": "normal",
    "benign": "normal",
    "normal.": "normal",
    "benign.": "normal",
    "attack": "dos",
    "dos": "dos",
    "ddos": "dos",
    "dos/ddos": "dos",
    "dos hulk": "dos",
    "dos goldeneye": "dos",
    "dos slowhttptest": "dos",
    "dos slowloris": "dos",
    "heartbleed": "dos",
    "neptune": "dos",
    "smurf": "dos",
    "back": "dos",
    "teardrop": "dos",
    "pod": "dos",
    "land": "dos",
    "probe": "probe",
    "probe/scan": "probe",
    "scan": "probe",
    "portscan": "probe",
    "portsweep": "probe",
    "ipsweep": "probe",
    "nmap": "probe",
    "satan": "probe",
    "r2l": "r2l",
    "ftp_patator": "r2l",
    "ssh_patator": "r2l",
    "guess_passwd": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "u2r": "u2r",
    "rootkit": "u2r",
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "xterm": "u2r",
    "malware": "malware",
    "ransomware": "malware",
    "bot": "malware",
    "infiltration": "malware",
    "web attack": "malware",
    "web attack brute force": "malware",
    "web attack xss": "malware",
    "web attack sql injection": "malware",
    "phishing": "phishing",
    "apt": "apt",
}


def normalize_label(label: object) -> str:
    normalized = str(label).lower().strip()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())

    if normalized in LABEL_ALIASES:
        return LABEL_ALIASES[normalized]

    for alias, canonical in LABEL_ALIASES.items():
        if alias and alias in normalized:
            return canonical

    return "normal"


def label_to_id(label: object) -> int:
    canonical = normalize_label(label)
    try:
        return CANONICAL_ATTACK_TYPES.index(canonical)
    except ValueError:
        return 0


def id_to_class_name(class_id: int) -> str:
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"Class_{class_id}"
