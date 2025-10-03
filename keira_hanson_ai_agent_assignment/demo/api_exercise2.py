import requests

def get_public_holidays(country_code="US", year=2024):
    """
    Get public holidays for a specific country and year
    Uses Nager.Date API (free, no key required)
    """
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raises an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {country_code}: {e}")
        return None


#  Main 
if __name__ == "__main__":
    countries = ['US', 'CA', 'GB']   # test with 3 countries
    summary = {}

    for country in countries:
        holidays = get_public_holidays(country)
        if holidays:
            print(f"\nPublic holidays in {country} (2024):")
            for holiday in holidays:
                name = holiday.get("localName", holiday.get("name", "Unknown"))
                date = holiday.get("date")
                print(f" - {date}: {name}")
            
            summary[country] = len(holidays)
        else:
            summary[country] = 0

    # Summary comparison
    print("\n==============================")
    print("Holiday Count Summary (2024)")
    print("==============================")
    for country, count in summary.items():
        print(f"{country}: {count} holidays")
