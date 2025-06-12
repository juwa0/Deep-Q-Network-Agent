import gymnasium as gym
import torch, random, cv2, numpy as np
from collections import deque
from dqn_agent import DQN, ReplayBuffer, DEVICE, empty_state

# ── hyper-params ────────────────────────────────────────────────────────────
EPISODES      = 500
BATCH_SIZE    = 32
GAMMA         = 0.99
LR            = 1e-4
EPS_START     = 1.0
EPS_END       = 0.1
EPS_DECAY     = 100_000          # steps
TARGET_SYNC   = 1_000            # steps
MEM_CAPACITY  = 100_000
STACK         = 4

# ── helpers ─────────────────────────────────────────────────────────────────
def preprocess(frame):
    gray   = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized= cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized

def stack_frames(deq, frame, first):
    f = preprocess(frame)
    if first:
        deq = deque([f]*STACK, maxlen=STACK)
    else:
        deq.append(f)
    return np.stack(deq, axis=0), deq

# ── main training loop ──────────────────────────────────────────────────────
def train():
    env = gym.make("CarRacing-v3", continuous=False)
    n_actions = env.action_space.n

    policy_net = DQN(n_actions).to(DEVICE)
    target_net = DQN(n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=LR)
    memory    = ReplayBuffer(MEM_CAPACITY)

    eps        = EPS_START
    global_step= 0
    rewards_log= []

    for ep in range(EPISODES):
        obs,_          = env.reset()
        state, stacker = stack_frames(None, obs, True)
        done, ep_reward= False, 0

        while not done:
            global_step += 1
            # ε-greedy action
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.tensor(state, device=DEVICE).unsqueeze(0)
                    action = int(policy_net(s).argmax())

            nxt_obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            nxt_state, stacker = stack_frames(stacker, nxt_obs, False)

            memory.push(state, action, r, nxt_state, done)
            state = nxt_state
            ep_reward += r

            # learn --------------------------------------------------------
            if len(memory) >= BATCH_SIZE:
                S,A,R,S2,D = memory.sample(BATCH_SIZE)
                q_vals = policy_net(S).gather(1, A.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next = target_net(S2).max(1)[0]
                    target   = R + GAMMA * max_next * (~D)
                loss = torch.nn.functional.mse_loss(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # sync target net
            if global_step % TARGET_SYNC == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # linear ε-decay
            eps = max(EPS_END, EPS_START - global_step / EPS_DECAY)

        rewards_log.append(ep_reward)
        print(f"Ep {ep:3d}  Reward {ep_reward:6.1f}  ε={eps:.3f}")

    env.close()
    policy_net.save("dqn_carracing.pth")

    # plot
    import matplotlib.pyplot as plt
    plt.plot(rewards_log)
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.title("DQN on CarRacing-v2")
    plt.savefig("reward_plot.png")
    plt.show()

if __name__ == "__main__":
    train()

