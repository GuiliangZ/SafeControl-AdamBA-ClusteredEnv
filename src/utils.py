'''
    Custom Replay buffer for MBPO
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import sparse
import osqp


def safetyindex(s):
    x = s[0]
    y = s[1]
    theta = s[2]
    x0 = 0.5
    y0 = 1.5
    margin = 0.2
    phi = (1 + 0.5 + margin) ** 2 - ((x0 - x) * np.sin(theta) - (y0 - y) * np.cos(theta)) ** 2
    return phi

# plot cricle
def obj(u, ustar):
    y = np.linalg.norm(u - ustar)
    return y

def circle(x, y, r):
    th = np.linspace(0, 2 * np.pi, 100)
    xunit = r * np.cos(th) + x
    yunit = r * np.sin(th) + y
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(x, y, color="darkred", linewidth=2)

def dphidx(s):
    x0 = 0.5
    y0 = 1.5
    x = s[0]
    y = s[1]
    theta = s[2]
    first_chain = -2 * ((x0-x)*np.sin(theta) - (y0-y)*np.cos(theta))
    dpds = [first_chain*(-np.sin(theta)),
            first_chain*np.cos(theta),
            first_chain*((x0-x)*np.cos(theta) + (y0-y)*np.sin(theta))]
    return dpds

def gx(s):
    theta = s[2]
    g = [np.cos(theta), 0,
         np.sin(theta), 0,
         0, 1]
    return g

def quadprog(H, f, A=None, b=None,
             initvals=None, verbose=True):

    qp_P = sparse.csc_matrix(H)
    qp_f = np.array(f)
    qp_l = -np.inf * np.ones(len(b))
    qp_A = sparse.csc_matrix(A)
    qp_u = np.array(b)
    model = osqp.OSQP()
    model.setup(P=qp_P, q=qp_f,
                A=qp_A, l=qp_l, u=qp_u, verbose=verbose)
    if initvals is not None:
        model.warm_start(x=initvals)
    results = model.solve()
    return results.x, results.info.status

class ReplayBuffer():
    '''
        Implementation of a simple Replay Buffer for TD3.
        You are free to modify this file, or implement your own replay buffer.
        However, notice that the provided starter code assumes
        implementation of this Replay Buffer.
        An important note: we store not_done, which is the negation of the
        done signal. not_done = 1 if done = 0, and vice versa.
        We also return not_done when sampling
    '''
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))


    def add(self, state, action, next_state, reward, done):
        '''
            adds a single transition tuple to the replay buffer.
            state: np array of size (state_dim, )
            action: np array of size (action_dim, )
            next_state: np array of size (state_dim, )
            reward: float
            done: float (1.0 if done, 0.0 otherwise)
        '''
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, state, action, next_state, reward, done):
        '''
            adds a batch of transition tuples to the replay buffer.
            state: np array of size (batch_size, state_dim)
            action: np array of size (batch_size, action_dim)
            next_state: np array of size (batch_size, state_dim)
            reward: np array of size (batch_size, )
            done: np array of size (batch_size, )
        '''
        batch_size = state.shape[0]
        if len(reward.shape) == 1:
            reward = reward.reshape((-1, 1))
        if len(done.shape) == 1:
            done = done.reshape((-1, 1))
        if batch_size >= self.max_size:
            self.state = state[batch_size - self.max_size:]
            self.action = action[batch_size - self.max_size:]
            self.next_state = next_state[batch_size - self.max_size:]
            self.reward = reward[batch_size - self.max_size:]
            self.not_done = (1. - done)[batch_size - self.max_size:]
            self.ptr = 0
            self.size = self.max_size
        # More than vacant space. Need to evict front
        elif batch_size >= self.max_size - self.size:
            vacancy = self.max_size - self.size
            remaining = batch_size - vacancy
            self.state[self.ptr:self.ptr + vacancy] = state[:vacancy]
            self.action[self.ptr:self.ptr + vacancy] = action[:vacancy]
            self.next_state[self.ptr:self.ptr + vacancy] = next_state[:vacancy]
            self.reward[self.ptr:self.ptr + vacancy] = reward[:vacancy]
            self.not_done[self.ptr:self.ptr + vacancy] = (1. - done)[:vacancy]
            self.ptr = 0
            self.state[:remaining] = state[vacancy:]
            self.action[:remaining] = action[vacancy:]
            self.next_state[:remaining] = next_state[vacancy:]
            self.reward[:remaining] = reward[vacancy:]
            self.not_done[:remaining] = (1. - done)[vacancy:]
            self.ptr += remaining
            self.size = self.max_size
        else:
            vacancy = batch_size
            self.state[self.ptr:self.ptr + vacancy] = state[:vacancy]
            self.action[self.ptr:self.ptr + vacancy] = action[:vacancy]
            self.next_state[self.ptr:self.ptr + vacancy] = next_state[:vacancy]
            self.reward[self.ptr:self.ptr + vacancy] = reward[:vacancy]
            self.not_done[self.ptr:self.ptr + vacancy] = (1. - done)[:vacancy]
            self.ptr += vacancy
            self.size += vacancy
        assert self.ptr < self.max_size


    def sample(self, batch_size):
        '''
            Samples a batch of transitions, with specified batch_size
            return them as float32 tf tensors.
        '''
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            tf.convert_to_tensor(self.state[ind].astype("float32")),
            tf.convert_to_tensor(self.action[ind].astype("float32")),
            tf.convert_to_tensor(self.next_state[ind].astype("float32")),
            tf.convert_to_tensor(self.reward[ind].astype("float32")),
            tf.convert_to_tensor(self.not_done[ind].astype("float32"))
        )

    def shuffle(self):
        '''
            Shuffles the replay buffer, without changing its contents.
        '''
        combined = np.hstack([self.state, self.action, self.next_state, self.reward, self.not_done])
        np.random.shuffle(combined)
        self.state = combined[:, :self.state_dim]
        self.action = combined[:, self.state_dim:self.state_dim + self.action_dim]
        self.next_state = combined[:, self.state_dim + self.action_dim:-2]
        self.reward = combined[:, -2:-1]
        self.not_done = combined[:, -1:]

    def sample_all(self):
        '''
            Sample all transitions in the replay buffer
            return them as float32 tf tensors.
        '''
        return (
            tf.convert_to_tensor(self.state[:self.size].astype("float32")),
            tf.convert_to_tensor(self.action[:self.size].astype("float32")),
            tf.convert_to_tensor(self.next_state[:self.size].astype("float32")),
            tf.convert_to_tensor(self.reward[:self.size].astype("float32")),
            tf.convert_to_tensor(self.not_done[:self.size].astype("float32"))
        )
