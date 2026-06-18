"""Map a national team name to a flag emoji (best effort; unknown -> "").

Flag emojis are two "regional indicator" letters built from a country's ISO
alpha-2 code. A few teams (England/Scotland/Wales) use special tag sequences.
"""

# Team name -> ISO alpha-2 code (covers the 2026 finalists + other common teams).
_ISO2 = {
    "Algeria": "DZ", "Argentina": "AR", "Australia": "AU", "Austria": "AT",
    "Belgium": "BE", "Bolivia": "BO", "Bosnia and Herzegovina": "BA", "Brazil": "BR",
    "Cameroon": "CM", "Canada": "CA", "Cape Verde": "CV", "Chile": "CL",
    "Colombia": "CO", "Costa Rica": "CR", "Croatia": "HR", "Curaçao": "CW",
    "Czech Republic": "CZ", "Denmark": "DK", "DR Congo": "CD", "Ecuador": "EC",
    "Egypt": "EG", "Finland": "FI", "France": "FR", "Germany": "DE", "Ghana": "GH",
    "Greece": "GR", "Haiti": "HT", "Hungary": "HU", "Iceland": "IS", "India": "IN",
    "Iran": "IR", "Iraq": "IQ", "Italy": "IT", "Ivory Coast": "CI", "Jamaica": "JM",
    "Japan": "JP", "Jordan": "JO", "Mexico": "MX", "Morocco": "MA", "Netherlands": "NL",
    "New Zealand": "NZ", "Nigeria": "NG", "Norway": "NO", "Panama": "PA",
    "Paraguay": "PY", "Peru": "PE", "Poland": "PL", "Portugal": "PT", "Qatar": "QA",
    "Republic of Ireland": "IE", "Romania": "RO", "Russia": "RU", "Saudi Arabia": "SA",
    "Senegal": "SN", "Serbia": "RS", "Slovakia": "SK", "Slovenia": "SI",
    "South Africa": "ZA", "South Korea": "KR", "Spain": "ES", "Sweden": "SE",
    "Switzerland": "CH", "Tunisia": "TN", "Turkey": "TR", "Ukraine": "UA",
    "United States": "US", "Uruguay": "UY", "Uzbekistan": "UZ", "Venezuela": "VE",
}

_SPECIAL = {
    "England": "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F",
    "Scotland": "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F",
    "Wales": "\U0001F3F4\U000E0067\U000E0062\U000E0077\U000E006C\U000E0073\U000E007F",
}


def flag(team):
    """Return the flag emoji for a team name, or '' if we don't have one."""
    if team in _SPECIAL:
        return _SPECIAL[team]
    code = _ISO2.get(team)
    if not code:
        return ""
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in code)


def with_flag(team):
    """'🇧🇷 Brazil' if we have a flag, else just 'Brazil'."""
    f = flag(team)
    return f"{f} {team}" if f else team
