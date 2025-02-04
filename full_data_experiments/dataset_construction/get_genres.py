import requests
import base64
import pandas as pd
# Replace these with your actual client_id and client_secret
client_id = 'cfe669d56fb2464a96ff4f68c529c78c'
client_secret = 'c60ca871d8d74f65b6c8e6bfe2fe1a45'

# Function to get access token
def get_access_token(client_id, client_secret):
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        'Authorization': f'Basic {auth_header}',
    }
    data = {
        'grant_type': 'client_credentials'
    }
    response = requests.post(auth_url, headers=headers, data=data)
    response_data = response.json()
    return response_data['access_token']

# Function to get artist genres by name
def get_artist_genres(artist_name, access_token):
    search_url = 'https://api.spotify.com/v1/search'
    headers = {
        'Authorization': f'Bearer {access_token}',
    }
    params = {
        'q': artist_name,
        'type': 'artist',
        'limit': 1
    }
    response = requests.get(search_url, headers=headers, params=params)
    response_data = response.json()
    if response_data['artists']['items']:
        artist = response_data['artists']['items'][0]
        return artist['genres']
    else:
        return None
import time

from copy import deepcopy
def update_genres(filename):

    df_fixed = pd.read_csv(filename)
    df_flex = deepcopy(df_fixed)

    for i in range(len(df_fixed)):
        artist_name = df_fixed.loc[i, 'artist_name']
        print(f"Processing artist {i}-th:{artist_name}...")

        if isinstance(df_fixed.loc[i, 'genres'], str):
            print(f"Artist {artist_name} already existed as a list....")
            genres = eval(df_fixed.loc[i, 'genres']) # Transformed into a list
            if len(genres) > 0: # If genres are already stored, skip
                print(f"Genres already stored for {artist_name}....")
            else:
                print(f"Genres missing for {artist_name}. We need to fetch them....")
                
            continue

        
        if pd.isna(df_fixed.loc[i, 'genres']):
            print(f"Artist {artist_name} is missing....")
            genres = [] # Empty list

            print(f"Genres not stored for {artist_name}. We need to fetch them....")
            found = False
            counter = 0
            while not found:
                print(f"Attempt {counter}....")
                time.sleep(0.5)
                artist_name = df_fixed.loc[i, 'artist_name']
                access_token = get_access_token(client_id, client_secret)
                genres = get_artist_genres(artist_name, access_token)
                print(artist_name)
                

                if genres is None:
                    genres = []

                if len(genres) > 0:
                    found = True
                    print(f"Success....")
                
                counter += 1
                if counter > 10:
                    print(f"Failure....")
                    break

            df_flex['genres'][i] = genres
            df_flex.to_csv(filename, index=False)

               

if __name__ == '__main__':
    update_genres(filename = 'olga_augmented_labels_.csv')