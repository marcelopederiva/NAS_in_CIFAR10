import numpy as np
from nas_cifar import training_n_out, gets_model, evaluate_in_test
from tqdm import tqdm

def in_bounds(point, bounds):
	# enumerate all dimensions of the point
	for d in range(len(bounds)):
		# check if out of bounds for this dimension
		if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
			return False
	return True

def evolutionary(bounds, n_iter, step_size, mu, lam):
    best, best_eval = None, 1e+10
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = []
    
    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])/2
            
        population.append(candidate)

	# perform the search
    the_bests=[]
    for epoch in range(n_iter):
        # evaluate fitness for the population
        scores = []
        for c in tqdm(range(len(population))):
             scores.append(training_n_out(epochs=2,
                                    batch_size=64,
                                    channel1=round(population[c][0]),
                                    channel2=round(population[c][1]),
                                    Kernel1=round(population[c][2]),
                                    Kernel2=round(population[c][3]),
                                    dropout=np.round(population[c][4],decimals=2)))

        # Minimize the Score
        # rank scores in ascending order

        ranks = np.argsort(np.argsort(scores))

        # select the indexes for the top mu ranked solutions
        selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
        # create children from parents
        children = []
        for i in selected:
            # check if this parent is the best solution ever seen CHECK IF IS MAX OR MIN
            if scores[i] > best_eval:
                best, best_eval = population[i], scores[i]
                print('\n%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
                print('----------------------------\n')
                moment_model = gets_model(epochs=2,
                                            batch_size=1,
                                            channel1=round(best[0]),
                                            channel2=round(best[1]),
                                            Kernel1=round(best[2]),
                                            Kernel2=round(best[3]),
                                            dropout=np.round(best[4],decimals=2))
                
                trainable_count = moment_model.count_params()
                the_bests.append([best, best_eval, trainable_count])

            # create children for parent
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    child = population[i] + np.multiply(np.random.randn(len(bounds)) , step_size)
                children.append(child)
        # replace population with children
        population = children
        best_model = gets_model(epochs=2,
                                batch_size=64,
                                channel1=round(best[0]),
                                channel2=round(best[1]),
                                Kernel1=round(best[2]),
                                Kernel2=round(best[3]),
                                dropout=np.round(best[4],decimals=2))
        
    return best, best_eval, the_bests, best_model





if __name__=='__main__':
    # Define range for input
    parameters = {'channel1' : [8,128],
                  'channel2' : [8,128],
                  'kernel1' :  [1,6],
                  'kernel2' :  [1,6],
                  'dropout' :  [0,1]}
    
    bounds = np.array([parameters['channel1'],
                       parameters['channel2'],
                       parameters['kernel1'],
                       parameters['kernel2'],
                       parameters['dropout'],
                       ])
    # Define the total iterations
    n_iter = 20
    # Number of parents selected
    mu = 3
    # The number of population
    lam = 15

    # Define the maximum step size
    step_size = [16,
                 16,
                 2,
                 2,
                 0.5]
    
    # Perform the evolution strategy (mu, lambda) search
    best, score, the_bests, best_model = evolutionary(bounds, n_iter, step_size, mu, lam)

    print('\n-----------------------------------------\n')
    print('Best Score\n')
    final_score = evaluate_in_test(epochs=15,
                                batch_size=64,
                                channel1=round(best[0]),
                                channel2=round(best[1]),
                                Kernel1=round(best[2]),
                                Kernel2=round(best[3]),
                                dropout=np.round(best[4],decimals=2))
    best_model.summary()

    print('f(%s) = %f' % (best, score))