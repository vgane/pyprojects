# This is a sample Python script.



import tensorflow_probability as tfp
# import tensorflow as tf



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    tfd = tfp.distributions
    initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
    transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                                     [0.2, 0.8]])

    observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])
    # model = tfd.HiddenMarkovModel(
    #     initial_distribution=initial_distribution,
    #     transition_distribution=transition_distribution,
    #     observation_distribution=observation_distribution,
    #     num_steps=7)
    # #
    # # The expected temperatures for each day are given by:
    #
    # model.mean()  # shape [7], elements approach 9.0
    #
    # # The log pdf of a week of temperature 0 is:
    #
    # model.log_prob(tf.zeros(shape=[7]))

if __name__ == '__main__':
    print_hi('PyCharm')

    # Feature columns describe how to use the input.
