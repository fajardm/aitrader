import json

result = []
with open("issi.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].isupper() and len(parts[0]) <= 6:
            symbol = parts[0]+".JK"
            name = " ".join(parts[1:])
            result.append({
                "symbol": symbol,
                "name": name,
                "ema_short": None,
                "ema_long": None,
                "rsi_period": None
            })

with open("issi.json", "w", encoding="utf-8") as out:
    json.dump(result, out, indent=2, ensure_ascii=False)

print(f"Saved {len(result)} tickers to issi.json")