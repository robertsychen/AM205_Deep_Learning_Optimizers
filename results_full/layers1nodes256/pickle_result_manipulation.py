import cPickle as pickle
import numpy as np

#new naming:
#
'''
results = []

#for number_name in [0,1,2,3,4,5,6,7,10,11,12,15,16,17,20,21,22,25,26,27]:
for number_name in xrange(9):
    result = pickle.load(open("new_new_result" + str(number_name) + ".pkl", "rb"))
    print
    print number_name
    print
    print result
    results.append(result)
    '''

#setting up master results dictionary
results_master = {}
for algo in ['gd','adam','hess','cg','bfgs']:
    results_master[algo] = {}
    for noise in ['none', 'test', 'both']:
        results_master[algo][noise] = {}
        for stat in ['time', 'acc', 'iters']:
            results_master[algo][noise][stat] = {}
            results_master[algo][noise][stat]['full'] = []
            results_master[algo][noise][stat]['mean'] = None
            results_master[algo][noise][stat]['sd'] = None

#loading robert's data format
for algo in ['gd','adam']:
    for noise in ['none', 'test', 'both']:
        result1 = pickle.load(open(algo + '_' + noise + '_' + str(1) + ".pkl", "rb"))
        result2 = pickle.load(open(algo + '_' + noise + '_' + str(2) + ".pkl", "rb"))
        results_master[algo][noise]['time']['full'] = result1[0][0] + result2[0][0]
        results_master[algo][noise]['acc']['full'] = result1[1][0] + result2[1][0]
        results_master[algo][noise]['iters']['full'] = result1[2][0] + result2[2][0]

#loading alex's data format
for algo in ['hess','cg','bfgs']:
    for noise in ['none', 'test', 'both']:
        result1 = pickle.load(open(algo + '_' + noise + '_' + str(1) + ".pkl", "rb"))
        result2 = pickle.load(open(algo + '_' + noise + '_' + str(2) + ".pkl", "rb"))
        results_master[algo][noise]['time']['full'] = result1['result'][0][0] + result2['result'][0][0]
        results_master[algo][noise]['acc']['full'] = result1['result'][1][0] + result2['result'][1][0]
        results_master[algo][noise]['iters']['full'] = result1['result'][2][0] + result2['result'][2][0]

#drop problematic cases of hessian-free
for stat in ['time', 'acc', 'iters']:
    results_master['hess']['none'][stat]['full'] = [results_master['hess']['none'][stat]['full'][i] for i in [0,1,2,4,5,6,8,9,10,12,13,14]]
    results_master['hess']['test'][stat]['full'] = [results_master['hess']['test'][stat]['full'][i] for i in [0,1,2,3,4,5,7,9,10,11,12,14]]
    results_master['hess']['both'][stat]['full'] = [results_master['hess']['both'][stat]['full'][i] for i in [1,2,3,4,5,6,9,11,12,13,14]]

#computing mean and SD
for algo in ['gd','adam','hess','cg','bfgs']:
    for noise in ['none', 'test', 'both']:
        for stat in ['time', 'acc', 'iters']:
            results_master[algo][noise][stat]['mean'] = np.asarray(results_master[algo][noise][stat]['full']).mean()
            results_master[algo][noise][stat]['sd'] = np.asarray(results_master[algo][noise][stat]['full']).std()

#save results
pickle.dump(results_master, open("master_results_dict.pkl", "wb"))

#print results nicely
for stat in ['time', 'acc', 'iters']:
    for algo in ['gd','adam','hess','cg','bfgs']:
        for noise in ['none', 'test', 'both']:
            print
            print algo, noise, stat
            print "All runs:"
            print [round(elem, 1) for elem in results_master[algo][noise][stat]['full']]
            print "Mean:"
            print round(results_master[algo][noise][stat]['mean'], 1)
            print "SD:"
            print round(results_master[algo][noise][stat]['sd'] / np.sqrt(15.0), 2)



