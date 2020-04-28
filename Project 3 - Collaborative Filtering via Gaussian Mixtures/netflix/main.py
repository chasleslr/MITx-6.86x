import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")


# -----------------------------------
# K-Means Algorithm
# -----------------------------------

print('\n----- K-Means Algorithm -----\n')

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

    print(f'K={k}', f'Best seed: {best_seed}', f'Cost: {cost}')
    #common.plot(X, mixture, post, title=f"K-Means, K={k}")


# -----------------------------------
# Expectation-Maximization Algorithm
# -----------------------------------

print('\n----- Expectation-Maximization Algorithm -----\n')

seeds = [0, 1, 2, 3, 4]
K = [1, 2, 3, 4]
bic = np.zeros(len(K))

for j, k in enumerate(K):
    mixtures = []
    posts = []
    logloss = np.empty(len(seeds))

    for i, seed in enumerate(seeds):
        # initialize mixture model with random points
        mixture, post = common.init(X, K=k, seed=seed)

        # run EM-algorithm
        mixture, post, LL = naive_em.run(X, mixture=mixture, post=post)

        mixtures.append(mixture)
        posts.append(post)
        logloss[i] = LL
        #print('K=', k, 'seed=', seed, 'logloss=', LL)

    best_seed = np.argmax(logloss)
    logloss = logloss[best_seed]
    mixture = mixtures[best_seed]
    post = posts[best_seed]

    current_bic = common.bic(X, mixture, logloss)
    bic[j] = current_bic

    print(f'K={k}', f'Best seed={best_seed}', f'logloss={logloss}', f'BIC={current_bic}')
    #common.plot(X, mixture, post, title=f"Naive-EM, K={k}")

best_K_ix = np.argmax(bic)
best_K = K[best_K_ix]
best_bic = bic[best_K_ix]
print(f"Best K={best_K}", f"BIC={best_bic}")


# -----------------------------------
# EM Algorithm for Matrix Completion
# -----------------------------------

print('\n Expectation-Maximization Algorithm for Matrix Completion')

X = np.loadtxt("netflix_incomplete.txt")

seeds = [0, 1, 2, 3, 4]
K = [1, 12]
bic = np.zeros(len(K))

for j, k in enumerate(K):
    mixtures = []
    posts = []
    logloss = np.empty(len(seeds))

    for i, seed in enumerate(seeds):
        # initialize mixture model with random points
        mixture, post = common.init(X, K=k, seed=seed)

        # run EM-algorithm
        mixture, post, LL = em.run(X, mixture=mixture, post=post)

        mixtures.append(mixture)
        posts.append(post)
        logloss[i] = LL
        print('K=', k, 'seed=', seed, 'logloss=', LL)

    best_seed = np.argmax(logloss)
    logloss = logloss[best_seed]
    mixture = mixtures[best_seed]
    post = posts[best_seed]

    current_bic = common.bic(X, mixture, logloss)
    bic[j] = current_bic

    print(f'K={k}', f'Best seed={best_seed}', f'logloss={logloss}', f'BIC={current_bic}')
    #common.plot(X, mixture, post, title=f"EM, K={k}")

best_K_ix = np.argmax(bic)
best_K = K[best_K_ix]
best_bic = bic[best_K_ix]
print(f"Best K={best_K}", f"BIC={best_bic}")