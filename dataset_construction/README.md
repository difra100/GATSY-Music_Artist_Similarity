# Oh, what a Large Graph of Artists---The OLGA Dataset

This repository contains the relevant meta-data and features of the OLGA dataset, which was used in the experiments in the following paper:

Filip Korzeniowski, Sergio Oramas, and Fabien Gouyon,  
**“Artist Similarity with Graph Neural Networks”**,  
in Proc. of the 22nd Int. Society for Music Information Retrieval Conf.,  
Online, 2021.

## Meta-Data

All metadata can be found in `olga.csv`.
It contains information about all 17,673 artists in the order as they appear in the feature files.
In particular, you can find the *musicbrainz_id* of the artist, the *partition* (train/val/test) they are in, and the musicbrainz ids of up to 25 of their *tracks* selected to compute their features.

## Features

`acousticbrainz.npy` provides, for each artist, the centroid of the AcousticBrainz features of their selected tracks. See the paper for details.

## Ground Truth

The ground truth can be obtained from AllMusic. To this end, you will first need to map the MusicBrainz artist ids to AllMusic ids. This can be done using the relationships provided by MusicBrainz itself, using their API. **Be aware of the API limitations MB enforces.** 

For example, for MusicBrainz artist `0c3319f3-d4c9-48d4-ba4c-deb7cd3c125b`, you can obtain a JSON string with the relevant links using this MusicBrainz API link:
```
https://musicbrainz.org/ws/2/artist/0c3319f3-d4c9-48d4-ba4c-deb7cd3c125b?inc=url-rels&fmt=json
```
In Python, you might want to use something like this:
```python
def get_mapping(musicbrainz_id):
    response = requests.get(f'https://musicbrainz.org/ws/2/artist/{musicbrainz_id}?inc=url-rels&fmt=json')
    
    if response.ok:
        data = response.json()
        refs = [r['url']['resource'] for r in data['relations'] if r['type'] == 'allmusic']        
        return refs[0] if len(refs) != 0 else None
    
    return None
```    

Given the AllMusic IDs, the lists of similar artists (as listed under "Similar To" in the "Related" tab in AllMusic's website) can be licensed from AllMusic.
