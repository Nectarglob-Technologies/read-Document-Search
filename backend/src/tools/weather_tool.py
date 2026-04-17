import requests


class WeatherTool:

    def get_weather(self, city):

        print(f"Fetching weather data for city: {city}")

        # 🔥 OPTION 1: FREE API (NO KEY)
        url = f"https://api.open-meteo.com/v1/forecast?latitude=19.07&longitude=72.87&current_weather=true"

        try:
            res = requests.get(url, timeout=5).json()

            current = res.get("current_weather", {})

            if not current:
                raise Exception("No data")

            return f"""
                City: {city}
                Temperature: {current.get('temperature')} °C
                Wind Speed: {current.get('windspeed')} km/h
                """

        except:
            print("Free API failed, trying fallback...")

        # 🔥 OPTION 2: WEATHERAPI (REQUIRES KEY)
        try:
            url = f"http://api.weatherapi.com/v1/current.json?key=YOUR_KEY&q={city}"
            res = requests.get(url, timeout=5).json()

            return f"""
                City: {city}
                Temperature: {res['current']['temp_c']} °C
                Condition: {res['current']['condition']['text']}
                Humidity: {res['current']['humidity']}
                """

        except:
            return "Weather data not available"
