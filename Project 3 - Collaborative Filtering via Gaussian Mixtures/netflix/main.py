import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")


# K-Means Algorithm

seeds = [0, 1, 2, 3, 4]
K = [1, 2, 3, 4]

for k in K:
    mixtures = []
    posts = []
    costs = np.empty(len(seeds))

    for i, seed in enumerate(seeds):
        # initialize mixture model with random points
        mixture, post = common.init(X, K=k, seed=seed)

        # run k-means
        mixture, post, cost = kmeans.run(X, mixture=mixture, post=post)

        mixtures.append(mixture)
        posts.append(post)
        costs[i] = cost

    best_seed = np.argmin(costs)
    cost = costs[best_seed]
    mixture = mixtures[best_seed]
    post = posts[best_seed]

    print(f'K={k}')
    print(f'Cost: {cost}')
    common.plot(X, mixture, post, title=f"K-Means, K={k}")
