import googlemaps
# Initialize the Google Maps client with your API key
gmaps = googlemaps.Client(key='YOUR_GOOGLE_MAPS_API_KEY')

def find_no_website_gems(query, location, min_rating=4.5, min_reviews=10):
    """
    Search Google Maps for businesses matching a query/location,
    filtering for high ratings and no website.
    """
    results = []
    
    # Perform the text search
    # Note: Google caps standard text search results per query; 
    # for broad US coverage, iterate through specific zip codes or cities.
    response = gmaps.places(query=query, location=location, radius=50000) # 50km radius
    
    for place in response.get('results', []):
        place_id = place.get('place_id')
        rating = place.get('rating', 0)
        user_ratings_total = place.get('user_ratings_total', 0)
        
        # Fetch detailed place information to check for website field
        details = gmaps.place(place_id=place_id, fields=['name', 'formatted_address', 'website', 'rating', 'user_ratings_total', 'formatted_phone_number'])
        result_data = details.get('result', {})
        
        has_website = 'website' in result_data
        
        # Filter criteria: Very good reviews (rating >= min_rating) and NO website
        if not has_website and rating >= min_rating and user_ratings_total >= min_reviews:
            results.append({
                'name': result_data.get('name'),
                'address': result_data.get('formatted_address'),
                'phone': result_data.get('formatted_phone_number', 'N/A'),
                'rating': rating,
                'total_reviews': user_ratings_total
            })
            
    return results

# Example usage: Search for plumbers in a specific US zip code area
us_location_coords = (37.7749, -122.4194) # Example: San Francisco area coordinates
businesses = find_no_website_gems(query='plumber', location=us_location_coords, min_rating=4.6, min_reviews=20)

for b in businesses:
    print(f"Found: {b['name']} | Rating: {b['rating']} ({b['total_reviews']} reviews) | Phone: {b['phone']}")

import requests

payload = { 'api_key': '7969ccca4d264055419af72f2c848e00', 'url': 'https://httpbin.org/' }
r = requests.get('https://api.scraperapi.com/', params=payload)
print(r.text)
