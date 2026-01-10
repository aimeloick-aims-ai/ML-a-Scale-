import csv

def load_features(data_by_movie, movieid_to_idx):
    genres_to_idx = {}
    idx_to_genre = []
    item_num = len(data_by_movie)
    item_features = [[] for _ in range(item_num)]
    features_having_items = []

    with open('ml-32m/movies.csv', mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            movie_id = int(row[0])
            if movie_id in movieid_to_idx:
                movie_index = movieid_to_idx[movie_id]
                genres = row[2]
                for genre in genres.split('|'):
                    if genre not in genres_to_idx:
                        genre_index = len(genres_to_idx)
                        idx_to_genre.append(genre)
                        genres_to_idx[genre] = genre_index
                        # Initialize a new list for this new genre in features_having_items
                        features_having_items.append([])
                    else:
                        genre_index = genres_to_idx[genre]

                    item_features[movie_index].append(genre_index)
                    features_having_items[genre_index].append(movie_index)
    return  item_features, features_having_items