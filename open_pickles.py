import cPickle as pickle

results = []

for number_name in xrange(12):
    result = pickle.load(open("newresult" + str(number_name) + ".pkl", "rb"))
    print
    print number_name
    print
    print result
    results.append(result)
