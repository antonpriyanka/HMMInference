import sys
import numpy as np
import matplotlib.pyplot as plt
from hmm import HMM

"""FUNCTIONS FOR FILE PROCESSING"""


def load_model(filename):
    model = {}
    input = open(filename, 'r')
    i = input.readline().split()
    model['states'] = i[0].split(",")

    input.readline()
    i = input.readline().split()
    x = i[0].split(",")
    model['initial'] = [float(i) for i in x]

    input.readline()
    tprob = []
    for state in range(len(model['states'])):
        i = input.readline().split()
        x = i[0].split(",")
        tprob.append([float(i) for i in x])
    model['tprob'] = tprob

    input.readline()
    i = input.readline().split()
    model['emissions'] = i[0].split(",")

    input.readline()
    eprob = []
    for state in range(len(model['states'])):
        i = input.readline().split()
        x = i[0].split(",")
        eprob.append([float(i) for i in x])
    model['eprob'] = eprob

    return model


def load_data(filename):
    input = open(filename, 'r')
    data = []
    for i in input.readlines():
        x = i.split()
        if x == [',']:
            y = [' ', ' ']
        else:
            y = x[0].split(",")
        data.append(y)
    observations = []
    classes = []
    for c, o in data:
        observations.append(o)
        classes.append(c)

    data = {'observations': observations, 'classes': classes}
    return data


"""FUNCTIONS FOR TESTING"""


def plot(beliefs, states, title):
    all_beliefs = np.array(beliefs)
    plt.plot(all_beliefs[:, 0], label=states[0], color='r', marker='o')
    plt.plot(all_beliefs[:, 1], label=states[1], color='b', marker='o')
    plt.plot(all_beliefs[:, 2], label=states[2], color='g', marker='o')
    plt.legend(loc='best')
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('probability')


def accuracy(a, b):
    return np.mean(np.core.defchararray.equal(a, b))


def test_filtering(hmm, observations):
    estimate = hmm.forward(observations)
    plot(estimate, hmm.states, 'Estimated belief states')
    print('\nFiltering - distribution over most recent state given short data set:')
    for i in range(0, len(hmm.states)):
        print('   ', hmm.states[i], '%1.3f' % estimate[-1, i])


def test_viterbi(hmm, type, observations, classes, short):
    best_sequence_short = [hmm.states[i] for i in hmm.viterbi(observations[0:short])]
    best_sequence_full = [hmm.states[i] for i in hmm.viterbi(observations)]

    if type == 'char':
        print('\nViterbi - predicted state sequence:   ', ''.join(best_sequence_short))
        print('Viterbi - actual state sequence:      ', ''.join(classes[0:short]))
    else:
        print('\nViterbi - predicted state sequence:\n   ', best_sequence_short)
        print('Viterbi - actual state sequence:\n   ', classes[0:short])

    print('\nThe accuracy of your Viterbi classifier on the short data set is',
          accuracy(classes[0:short], best_sequence_short))
    print('The accuracy of your Viterbi classifier on the entire data set is', accuracy(classes, best_sequence_full))


def test_smoothing(hmm, observations, short):
    smoothed_short = hmm.smooth(observations[0:short])
    smoothed_full = hmm.smooth(observations)
    plot(smoothed_full[0:short], hmm.states, 'Smoothed belief states')

    print('\nSmoothing - distribution over most recent state given short data set:')
    for i in range(0, len(smoothed_short[0])):
        print('   ', hmm.states[i], '%1.3f' % smoothed_short[short - 1, i])
    print('\nSmoothing - distribution over same state above given full data set:')
    for i in range(0, len(smoothed_full[0])):
        print('   ', hmm.states[i], '%1.3f' % smoothed_full[short - 1, i])


if __name__ == '__main__':
    weather_model = load_model('weather.model')
    weather_data = load_data('weather.data')
    weather_hmm = HMM(weather_model)
    weather_obs_indices = [weather_hmm.emissions.index(i) for i in weather_data['observations']]

    plt.figure(1)
    print('\nWeather observation sequence:\n   ', weather_data['observations'][0:10])
    test_filtering(weather_hmm, weather_obs_indices[0:50])
    test_viterbi(weather_hmm, 'weather', weather_obs_indices, weather_data['classes'], 10)

    plt.figure(2)
    test_smoothing(weather_hmm, weather_obs_indices, 50)
    plt.show()

    typo_model = load_model('typo.model')
    typo_data = load_data('typo.data')
    typo_hmm = HMM(typo_model)
    typo_obs_indices = [typo_hmm.emissions.index(i) for i in typo_data['observations']]
    print('\nTypo observation sequence:   ', ''.join(typo_data['observations'][0:84]))
    test_viterbi(typo_hmm, 'char', typo_obs_indices, typo_data['classes'], 84)
