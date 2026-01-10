import csv
# Unzip the downloaded file

def structure_data():
    userid_to_idx = {}
    idx_to_user =  []
    movieid_to_idx = {}
    idx_to_movie = []
    data_by_user = []
    data_by_movie = []

    with open('ml-32m/ratings.csv', mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:
            user_id = int(row[0])
            movie_id = int(row[1])
            #ranking_value = float(row[2])

            if user_id not in userid_to_idx:
                userid_to_idx[user_id] = len(userid_to_idx)
                idx_to_user.append(user_id)
                data_by_user.append([])

            if movie_id not in movieid_to_idx:
                movieid_to_idx[movie_id] = len(movieid_to_idx)
                idx_to_movie.append(movie_id)
                data_by_movie.append([])

            prov_user_key = userid_to_idx[user_id]
            prov_movie_key = movieid_to_idx[movie_id]
            data_by_user[prov_user_key].append((prov_movie_key,float(row[2])))
            data_by_movie[prov_movie_key].append((prov_user_key,float(row[2])))
    return idx_to_movie, idx_to_user, movieid_to_idx, userid_to_idx, data_by_movie, data_by_user