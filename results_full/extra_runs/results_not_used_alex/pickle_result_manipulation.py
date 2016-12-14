import cPickle as pickle


#new naming:
#

results = []

#for number_name in [0,1,2,3,4,5,6,7,10,11,12,15,16,17,20,21,22,25,26,27]:
for number_name in xrange(9):
    result = pickle.load(open("new_new_result" + str(number_name) + ".pkl", "rb"))
    print
    print number_name
    print
    print result
    results.append(result)