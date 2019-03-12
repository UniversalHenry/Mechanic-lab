import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

np.random.seed(1)

class envModel:
    def __init__(self, m=[800.0, 10.0, 200.0], c=[10.0, 100.0], k=[1e5, 0.0],
                 x=[0.0, 0.35, 2.0], v=[10.0, 0.0, 0.0], F=0.0, rec=True,
                 r=[0.25, 0.05, 0.15], t0=0.0, dt=1e-3, gamma=1.5):
        self.m, self.c, self.k, self.x, self.v, self.r, self.t0, self.dt, self.F, self.gamma, self.rec \
            = m, c, k, x, v, r, t0, dt, F, gamma, rec
        self.contact = (r[0] + r[1] > x[1] - x[0])
        if rec:
            self.info = {}
            self.info['t'] = [t0]
            self.info['x'] = [x]
            self.info['v'] = [v]
            a = self.func(x+v,t0)
            self.info['a'] = [a[3:6]]
            self.info['c'] = [c]
            self.info['contact'] = [self.contact]

    def set(self,c2 = 100.0, k2 = 0.0, F = 0.0, rec = True, dt = 1e-3):
        self.c[1] = c2
        self.k[1] = k2
        self.F = F
        self.rec = rec
        self.dt = dt
        print('c:',self.c,'\t|','k:',self.k,'\t|','F:',self.F,'\t|','dt:',self.dt,'\t|','t:',self.t0,)

    def gen(self):
        top_m, low_m = 2000.0, 500.0
        top_c1, low_c1 = 100.0, 10.0
        top_k1, low_k1 = 1e7, 1e6
        top_gamma, low_gamma = 2.4, 1.3
        top_v, low_v = 30, 5
        self.m[0] = np.random.random_sample() * (top_m - low_m) + low_m
        self.c[0] = np.random.random_sample() * (top_c1 - low_c1) + low_c1
        self.k[0] = np.random.random_sample() * (top_k1 - low_k1) + low_k1
        self.gamma = np.random.random_sample() * (top_gamma - low_gamma) + low_gamma
        self.r[0] = self.r[2] * ((self.m[0] / self.m[2]) ** (1 / 3))
        self.x = [0.0, 0.35, 2.0]
        self.v = [np.random.random_sample() * (top_v - low_v) + low_v, 0.0, 0.0]
        self.set()
        print('New envModel generated!')
        print('m:', self.m, '\t|', 'r:', self.r, '\t|', 'x:', self.x, '\t|', 'v:', self.v, '\t|', 'rec:', self.rec, )
        print('c:', self.c, '\t|', 'k:', self.k, '\t|', 'F:', self.F, '\t|', 'dt:', self.dt, '\t|', 't:', self.t0, )

    def func(self, xv, t):
        m, c, k, r, gamma, F = self.m, self.c, self.k, self.r, self.gamma, self.F
        x = xv[0:3]
        v = xv[3:6]
        self.contact = (self.r[0] + self.r[1] > x[1] - x[0])
        if self.contact:
            va = [v[0], v[1], v[2],
                  -(c[0] * (abs(v[0]) ** gamma) * np.sign(v[0]) - c[0] * (abs(v[1]) ** gamma) * np.sign(v[1])  +
                    k[0] * x[0] - k[0] * x[1] +
                    k[0] * r[0] + k[0] * r[1]
                    ) / m[0],
                  -(-c[0] * (abs(v[0]) ** gamma) * np.sign(v[0]) + c[0] * (abs(v[1]) ** gamma) * np.sign(v[1])+
                    c[1] * v[1] - c[1] * v[2] +
                    -k[0] * x[0] + (k[0] + k[1]) * x[1] - k[1] * x[2] +
                    -k[0] * r[0] - (k[0] - k[1]) * r[1] + k[1] * r[2]
                    ) / m[1],
                  -(-F +
                    -c[1] * v[1] + c[1] * v[2] +
                    -k[1] * x[1] + k[1] * x[2] +
                    -k[1] * r[1] - k[1] * r[2]
                    ) / m[2]
                  ]

        else:
            va = [v[0], v[1], v[2],0,
                  -(c[1] * v[1] - c[1] * v[2] +
                    k[1] * x[1] - k[1] * x[2] +
                    k[1] * r[1] + k[1] * r[2]
                    ) / m[1],
                  -(-F +
                    -c[1] * v[1] + c[1] * v[2] +
                    -k[1] * x[1] + k[1] * x[2] +
                    -k[1] * r[1] - k[1] * r[2]
                    ) / m[2]
                  ]
        return va

    def solve(self):
        t = self.t0 + self.dt
        tspan = np.linspace(self.t0,t,101)
        sol = odeint(self.func,self.x+self.v,tspan)
        xv = sol[-1].tolist()
        if self.rec:
            self.info['t'] += [t]
            self.info['x'] += [xv[0:3]]
            self.info['v'] += [xv[3:6]]
            a = self.func(xv, t)
            self.info['a'] += [a[3:6]]
            self.info['c'] += [self.c]
            self.info['contact'] += [self.contact]
        self.x = xv[0:3]
        self.v = xv[3:6]
        self.t0 = t

if __name__ == '__main__':
    EnvModel = envModel()
    while EnvModel.t0 < 20:
        EnvModel.solve()
    v = np.array(EnvModel.info['v'])
    t = np.array(EnvModel.info['t'])
    plt.plot(t, v[:, 0], 'r', label='v1')
    plt.plot(t, v[:, 1], 'g', label='v2')
    plt.plot(t, v[:, 2], 'b', label='v3')
    plt.xlabel('t')
    plt.grid()
    plt.legend()
    plt.show()
