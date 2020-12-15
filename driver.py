import numpy as np
import matplotlib.pyplot as plt
import gym
from actorCritic import Agent
from actorCritic import ActorCritic

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    gamma = 0.92
    alpha = 0.0005
    layer1Dims = 512
    layer2Dims = 128
    layer3Dims = 128
    agent1Layer = Agent(gamma = gamma, alpha = alpha, inputDims = 4, numActions = 2, layer1Dims = layer1Dims)
    agent2Layer = Agent(gamma = gamma, alpha = alpha, inputDims = 4, numActions = 2, layer1Dims = layer1Dims, layer2Dims = layer2Dims)
    agent3Layer = Agent(gamma = gamma, alpha = alpha, inputDims = 4, numActions = 2, layer1Dims = layer1Dims, layer2Dims = layer2Dims, layer3Dims = layer3Dims)

    numEpisodes = 500

    fname = f"FinalActorCritic_{alpha}_{gamma}_{layer1Dims}_{layer2Dims}_{layer3Dims}"
    figure_file =  fname + '.png'

    #### TRAINING AGENT WITH 1 HIDDEN LAYER
    episodes1Layer = np.zeros((numEpisodes, 2))
    for i in range(numEpisodes):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent1Layer.chooseAction(observation)
            newObservation, reward, done, info = env.step(action)
            score += reward
            agent1Layer.learn(observation, reward, newObservation, done)
            observation = newObservation
        episodes1Layer[i, 0] = i
        episodes1Layer[i, 1] = score
        print(f"agent: 1 Layer - \t episode: {i},\t score {score}")

    #### TRAINING AGENT WITH 2 HIDDEN LAYERS
    episodes2Layer = np.zeros((numEpisodes, 2))
    for i in range(numEpisodes):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent2Layer.chooseAction(observation)
            newObservation, reward, done, info = env.step(action)
            score += reward
            agent2Layer.learn(observation, reward, newObservation, done)
            observation = newObservation
        episodes2Layer[i, 0] = i
        episodes2Layer[i, 1] = score
        print(f"agent: 2 Layer - \t episode: {i},\t score {score}")

    #### TRAINING AGENT WITH 3 HIDDEN LAYERS
    episodes3Layer = np.zeros((numEpisodes, 2))
    for i in range(numEpisodes):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent3Layer.chooseAction(observation)
            newObservation, reward, done, info = env.step(action)
            score += reward
            agent3Layer.learn(observation, reward, newObservation, done)
            observation = newObservation
        episodes3Layer[i, 0] = i
        episodes3Layer[i, 1] = score
        print(f"agent: 3 Layer - \t episode: {i},\t score {score}")

    env.close()

    #### CREATE RUNNING AVERAGES SO THE VALUES DON'T HAVE TOOOOO MUCH VARIANCE
    runningAvgs = np.zeros((numEpisodes,3))
    for i in range(len(runningAvgs)):
        print(i)
        runningAvgs[i,0] = np.mean(episodes1Layer[max(0, i-50):(i+1),1])
        runningAvgs[i,1] = np.mean(episodes2Layer[max(0, i-50):(i+1),1])
        runningAvgs[i,2] = np.mean(episodes3Layer[max(0, i-50):(i+1),1])

    #### GRAPH THE LINES
    plt.plot(episodes1Layer[:,0], runningAvgs[:,0], label = "1 Layer Score", color = "#0A284B")
    plt.plot(episodes2Layer[:,0], runningAvgs[:,1], label = "2 Layer Score", color = "#235FA4")
    plt.plot(episodes3Layer[:,0], runningAvgs[:,2], label = "3 Layer Score", color = "#A691AE")
    plt.title(f"Reward Growth over {numEpisodes} Episodes")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(figure_file)
