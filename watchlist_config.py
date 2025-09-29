"""
Watchlist Configuration with JSON Persistence
============================================
Configure default ticker symbols for the AI Trader UI
"""
import json
from typing import Dict, List

# Configuration file
WATCHLIST_FILE = "watchlists.json"

# Default watchlist configuration
DEFAULT_CONFIG = {
    "default_watchlist": "main",
    "watchlists": {
        "main": ["WIRG.JK", "BRMS.JK", "NCKL.JK", "EMTK.JK", "BTPS.JK", "CUAN.JK", "MDKA.JK", "IMPC.JK", "WIFI.JK", "ANTM.JK", "EMAS.JK", "BRPT.JK", "FILM.JK", "PTRO.JK"]
    }
}

def load_watchlists() -> Dict:
    """Load watchlists from JSON file"""
    try:
        with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Create default file if it doesn't exist
        save_watchlists(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

def save_watchlists(config: Dict):
    """Save watchlists to JSON file"""
    with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def get_watchlist(name="main") -> List[str]:
    """Get watchlist by name"""
    config = load_watchlists()
    if name is None:
        name = config.get("default_watchlist", "main")
    return config.get("watchlists", {}).get(name, DEFAULT_CONFIG["watchlists"]["main"])

def get_all_watchlists() -> Dict[str, List[str]]:
    """Get all available watchlists"""
    config = load_watchlists()
    return config.get("watchlists", DEFAULT_CONFIG["watchlists"])

def get_watchlist_names() -> List[str]:
    """Get list of watchlist names"""
    config = load_watchlists()
    return list(config.get("watchlists", {}).keys())

def add_watchlist(name: str, symbols: List[str]):
    """Add a new watchlist"""
    config = load_watchlists()
    config["watchlists"][name] = symbols
    save_watchlists(config)

def update_watchlist(name: str, symbols: List[str]):
    """Update an existing watchlist"""
    config = load_watchlists()
    if name in config["watchlists"]:
        config["watchlists"][name] = symbols
        save_watchlists(config)

def delete_watchlist(name: str):
    """Delete a watchlist"""
    config = load_watchlists()
    if name in config["watchlists"] and name != "main":  # Protect main watchlist
        del config["watchlists"][name]
        save_watchlists(config)

def set_default_watchlist(name: str):
    """Set the default watchlist"""
    config = load_watchlists()
    if name in config["watchlists"]:
        config["default_watchlist"] = name
        save_watchlists(config)

def add_to_watchlist(watchlist_name: str, symbols: List[str]):
    """Add symbols to an existing watchlist"""
    config = load_watchlists()
    if watchlist_name in config["watchlists"]:
        for symbol in symbols:
            if symbol not in config["watchlists"][watchlist_name]:
                config["watchlists"][watchlist_name].append(symbol)
        save_watchlists(config)

def remove_from_watchlist(watchlist_name: str, symbols: List[str]):
    """Remove symbols from a watchlist"""
    config = load_watchlists()
    if watchlist_name in config["watchlists"]:
        for symbol in symbols:
            if symbol in config["watchlists"][watchlist_name]:
                config["watchlists"][watchlist_name].remove(symbol)
        save_watchlists(config)