import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, jacobian

class PendulumWorld(object):
    def __init__(self, lengths, masses, num_pix=50, time_step=0.2, g=1.0, width=1.0):
        self.lengths = np.array(lengths)
        self.masses = np.array(masses)
        self.N = len(lengths) # number of arms
        self.acceleration = lagrangian_to_acceleration(self.lagrangian)
        self.num_pix, self.time_step, self.g = num_pix, time_step, g
        self.width = width

    def lagrangian(self, angles, omegas):
        y = np.cumsum(self.lengths * np.cos(angles))
        x_dot = np.cumsum( self.lengths * np.cos(angles) * omegas)
        y_dot = np.cumsum(-self.lengths * np.sin(angles) * omegas)
        V = np.sum(y * self.masses) * self.g
        T = 0.5 * np.sum(self.masses * (x_dot**2 + y_dot**2))
        return T - V

    def new_day(self):
        angles = npr.uniform(0, 2 * np.pi, self.N)
        omegas = npr.uniform(-1.0, 1.0, self.N)
        self.state = angles, omegas

    def next_state(self, action):
        angles, omegas = self.state
        self.state = leapfrog_integrator(angles, omegas, self.acceleration, self.time_step)
        return self.render()

    def render(self):
        low, high = 0.1, 0.9
        angles, _ = self.state
        canvas = np.zeros((self.num_pix, self.num_pix)) + low
        radius = np.sum(self.lengths)
        joint_coords_x = np.cumsum(self.lengths * np.sin(angles)) / radius / 1.2
        joint_coords_y = np.cumsum(self.lengths * np.cos(angles)) / radius / 1.2
        joint_coords_x = np.concatenate((np.zeros(1), joint_coords_x))
        joint_coords_y = np.concatenate((np.zeros(1), joint_coords_y))
        joint_coords = np.concatenate((joint_coords_x[:, None],
                                       joint_coords_y[:, None]), axis=1)
        canvas_coords = array_meshgrid(self.num_pix)
        for point_A, point_B in zip(joint_coords[:-1], joint_coords[1:]):
            D = distance_to_segment(point_A, point_B, canvas_coords)
            canvas = np.maximum(canvas, high * np.exp(-((D/self.width)*20)**4))
        return canvas

def lagrangian_to_acceleration(L):
    dL_dx = grad(L, argnum=0)
    p     = grad(L, argnum=1)
    dp_dx = jacobian(p, argnum=0)
    dp_dv = jacobian(p, argnum=1)
    def acceleration(x, v):
        return np.linalg.solve(dp_dv(x, v), dL_dx(x, v) - np.dot(dp_dx(x, v), v))

    return acceleration

def array_meshgrid(num_pix):
    # Creates (num_pix, num_pix, 2) array of x, y coords of (num_pix, num_pix) canvas
    pix_pos = np.linspace(-1, 1, num_pix)
    x = np.repeat(pix_pos[None,    :], num_pix, axis=0)
    y = np.repeat(pix_pos[::-1, None], num_pix, axis=1)
    return np.concatenate((x[:, :, None], y[:, :, None]), axis=2)

def distance_to_segment(A, B, X):
    # Shapes of A, B, X are (D,), (D,), (..., D) respectively
    # Computes squared distance of points X from line segment AB
    AB_length = np.linalg.norm(B - A)
    AB_hat = (B - A) / AB_length  # Unit vector from A to B
    s = np.dot(X - A, AB_hat)     # Distance along segment AB of closest point
    s_bounded = np.minimum(AB_length, np.maximum(0., s))
    closest_point_on_segment = A + s_bounded[..., None] * AB_hat
    return np.linalg.norm(X - closest_point_on_segment, axis=-1)

def leapfrog_integrator(x, v, a, t):
    N_steps = 1 # Should determine this adaptively
    dt = t / N_steps
    x += 0.5 * dt * v
    for i in range(N_steps - 1):
        v += dt * a(x, v)
        x += dt * v

    v += dt * a(x, v)
    x += 0.5 * dt * v
    return x, v
