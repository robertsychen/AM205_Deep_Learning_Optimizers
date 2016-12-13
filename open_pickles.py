import cPickle as pickle

results = []

for number_name in xrange(21):
    result = pickle.load(open("result" + str(number_name) + ".pkl", "rb"))
    print
    print number_name
    print
    print result
    results.append(result)
