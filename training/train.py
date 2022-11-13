import sys
sys.path.append("../util/")
import pongsimulation

# TODO multiplayer training, automated hyperparameter search, double q learning, n step q learning


'''
Main function: Defines important constants, initializes all the important classes and does the training.
'''

params = {"env_name": "RoboschoolPong-v1",
            "gamma": 0.99,  # discoutn factor in Bellman update
            "batch_size": 290,  # how many samples at the same time (has to be big for convergence of TD 1 step)
            "load_from": "../policy/best_train.dat",  # Set to filename as string if previous model should be loaded
            "replay_size": 7000,  # size of replay buffer
            "learning_rate": 1e-4,  # learning rate of neural network update
            "sync": 500,  # when to sync neural net and target network (low values destroy loss func)
            "eps_decay_rate": 10000,  # how fast does the epsilon exploration decay
            "training_frames": 1000000,  # total number of training frames
            "nactions": 2,  # network doesnt seem to care much about action_space discretization...
            "skip_frames": 4,  # how many frames are skipped with repeated actions != n step DQN
            "eps_start": 1,
            "eps_end": 0.02,
            "device": "cpu",
            "double": True,
            "selfplay": False,
            "player_n": 0,
            "messages_enabled": True,
            "mode": "comp", #comp for competition, coop for cooperation , trans for transition
            "nstep": 2}


if __name__ == '__main__':
    if params["selfplay"]:
        sim = pongsimulation.PongSelfplaySimulation(params)
    else:
        sim = pongsimulation.Simulation(params)

    sim.run("train")

