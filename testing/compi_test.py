import os, sys, subprocess
import numpy as np
import gym
import roboschool
import time
import gym.wrappers
import torch
# add path for util folder to import dqn and wrappers
sys.path.append("../util/")
from dqn_model import DQN, calc_loss
import wrappers as wrappers


def play(env, pi, video):
    episode_n = 0
    while 1:
        episode_n += 1
        obs = env.reset()
        if video: video_recorder = gym.monitoring.video_recorder.VideoRecorder(env=env, base_path=("/tmp/demo_pong_episode%i" % episode_n), enabled=True)
        while 1:
            a = pi.act(obs)
            obs, rew, done, info = env.step(a)
            if video: video_recorder.capture_frame()
            if done: break
        if video: video_recorder.close()
        break

def dqn_play(env,net, epsilon=0.0, device="cpu", mode = "comp"):
    """
    Epsilon greedy step. With probability epsilon, a random action is taken (exploration),
    else the action ist chosen to maximize the q-value as approximated by net (exploitation).
    """
    episode_n = 0
    while 1:
        episode_n += 1
        obs = env.reset()
        # video_recorder = gym.monitoring.video_recorder.VideoRecorder(env=env, base_path=("/tmp/demo_pong_episode%i" % episode_n), enabled=True)
        while 1:
            action = net(torch.tensor(obs, dtype=torch.float32)).max(0)[1]
            action = action.item()
            action = int(action)
            obs, rew, done, info = env.step(action)
            print(info)
            if done: break
        break

if len(sys.argv)==1:
    import roboschool.multiplayer
    game = roboschool.gym_pong.PongSceneMultiplayer()
    gameserver = roboschool.multiplayer.SharedMemoryServer(game, "pongdemo", want_test_window=True)
    for n in range(game.players_count):
        subprocess.Popen([sys.executable, sys.argv[0], "pongdemo", "%i"%n])
    gameserver.serve_forever()

else:
    player_n = int(sys.argv[2])

    env = gym.make("RoboschoolPong-v1")
    env.unwrapped.multiplayer(env, game_server_guid=sys.argv[1], player_n=player_n)

    # from Player1 import SmallReactivePolicy as Pol1
    # from Player2 import SmallReactivePolicy as Pol2
    if player_n==0:
        env = wrappers.action_space_discretizer(env, 2)
        net = DQN(env.observation_space.shape[0], env.action_space.n)
        net.load_state_dict(torch.load("../policy/best_train.dat"))
        dqn_play(env, net, epsilon=0.0, device="cpu", mode = "comp")
    else:
        # env = wrappers.action_space_discretizer(env, 2)
        # net = DQN(env.observation_space.shape[0], env.action_space.n)
        # net.load_state_dict(torch.load("../policy/best_train.dat"))
        # dqn_play(env, net, epsilon=0.0, device="cpu", mode = "comp")
        from Player1 import SmallReactivePolicy as Pol1
        pi = Pol1(env.observation_space, env.action_space)
        play(env,pi,video=False)
    # play(env, pi, video=False)   # set video = player_n==0 to record video

