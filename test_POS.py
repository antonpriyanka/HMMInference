import numpy as np
from hmm import HMM

"""FUNCTIONS FOR FILE PROCESSING"""


def load_tmodel(filename):
    tmp = np.loadtxt(filename, usecols=1, dtype=np.dtype('U'))
    states = np.unique(tmp).tolist()
    initial = np.zeros(len(states))
    tprob = np.zeros((len(states), len(states)))
    for line in open(filename, 'r').read().splitlines():
        x = line.split()
        if x[0] == '#':
            initial[states.index(x[1])] = x[2]
        else:
            tprob[states.index(x[0]), states.index(x[1])] = x[2]
    return {'states': states, 'initial': initial, 'tprob': tprob}


def load_emodel(filename, model):
    tmp = np.loadtxt(filename, usecols=1, dtype=np.dtype('U'))
    emissions = np.unique(tmp).tolist()
    states = model['states']
    eprob = np.zeros((len(states), len(emissions)))
    for line in open(filename, 'r').read().splitlines():
        x = line.split()
        eprob[states.index(x[0]), emissions.index(x[1])] = x[2]
    model['emissions'] = emissions
    model['eprob'] = eprob
    return model


def load_data(filename, model):
    lines = open(filename, 'r').read().splitlines()
    classes = [word for line in lines[0::2] for word in line.split()]
    observations = [word for line in lines[1::2] for word in line.split()]
    data = {'observations': observations, 'classes': classes}
    return data


"""FUNCTIONS FOR TESTING"""


def accuracy(a, b):
    return np.mean(np.core.defchararray.equal(a, b))


def test_viterbi(hmm, observations, classes, short):
    best_sequence_short = [hmm.states[i] for i in hmm.viterbi(observations[0:short])]
    best_sequence_full = [hmm.states[i] for i in hmm.viterbi(observations)]

    print('\nViterbi - predicted state sequence:\n   ', best_sequence_short)
    print('Viterbi - actual state sequence:\n   ', classes[0:short])
    print('\nThe accuracy of your Viterbi classifier on the short data set is',
          accuracy(classes[0:short], best_sequence_short))
    print('The accuracy of your Viterbi classifier on the entire data set is', accuracy(classes, best_sequence_full))


if __name__ == '__main__':
    model = load_tmodel('POS_transmission.model')
    model = load_emodel('POS_emission.model', model)
    data = load_data('POS.data', model)
    POS_hmm = HMM(model)
    obs_indices = [POS_hmm.emissions.index(i) for i in data['observations']]

    short = 37
    print('\nPOS observation sequence:\n   ', data['observations'][0:short])
    test_viterbi(POS_hmm, obs_indices, data['classes'], 37)
