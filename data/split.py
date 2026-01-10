import random


#data_by_movie, data_by_user = structure_data()
def data_split(data_by_movie, data_by_user, threshold = 0.9, rand = True):
    data_by_user_train = []
    data_by_user_test = []
    movie_number = len(data_by_movie)
    data_by_movie_train = [[] for _ in range(movie_number)]
    data_by_movie_test = [[] for _ in range(movie_number)]

    for user_idx, ratings in enumerate(data_by_user):
        if not ratings:
            continue
        if rand:
            random.shuffle(ratings)
        split_point = int(threshold * len(ratings))

        train_r = ratings[:split_point]
        test_r = ratings[split_point:]

        data_by_user_train.append(train_r)
        data_by_user_test.append(test_r)

        for movie_idx, rating in train_r:
            data_by_movie_train[movie_idx].append((user_idx, rating))
        for movie_idx, rating in test_r:
            data_by_movie_test[movie_idx].append((user_idx, rating))

    print(f"Train users: {len(data_by_user_train)}, Movies: {len(data_by_movie_train)}")
    print(f"Test users:  {len(data_by_user_test)}, Movies: {len(data_by_movie_test)}")
    return data_by_user_train, data_by_user_test,data_by_movie_train, data_by_movie_test



def data_split_standard(
    data_by_movie,
    data_by_user,
    threshold=0.9,
    rand=True
):
    movie_number = len(data_by_movie)
    user_number = len(data_by_user)


    users = [u for u, ratings in enumerate(data_by_user) if len(ratings) > 0]

    if rand:
        random.shuffle(users)

    split_point = int(threshold * len(users))
    train_users = set(users[:split_point])
    test_users  = set(users[split_point:])

    data_by_user_train  = [[] for _ in range(user_number)]
    data_by_user_test   = [[] for _ in range(user_number)]
    data_by_movie_train = [[] for _ in range(movie_number)]
    data_by_movie_test  = [[] for _ in range(movie_number)]

    for user_idx, ratings in enumerate(data_by_user):

        if user_idx in train_users:
            for movie_idx, rating in ratings:
                data_by_user_train[user_idx].append((movie_idx, rating))
                data_by_movie_train[movie_idx].append((user_idx, rating))

        elif user_idx in test_users:
            for movie_idx, rating in ratings:
                data_by_user_test[user_idx].append((movie_idx, rating))
                data_by_movie_test[movie_idx].append((user_idx, rating))

    n_train_inter = sum(len(u) for u in data_by_user_train)
    n_test_inter  = sum(len(u) for u in data_by_user_test)
    print(f"Train users:        {len(train_users)}")
    print(f"Test users:         {len(test_users)}")
    print(f"Train interactions:{n_train_inter}")
    print(f"Test interactions: {n_test_inter}")

    return (
        data_by_user_train,
        data_by_user_test,
        data_by_movie_train,
        data_by_movie_test
    )
