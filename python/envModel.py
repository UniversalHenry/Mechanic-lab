import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

np.random.seed(1)

class envModel:
    def __init__(self, m=[800.0, 10.0, 200.0], c=[10.0, 100.0], k=[1e10, 0.0],
                 x=[0.0, 0.35, 1000.0], v=[10.0, 0.0, 0.0], F=0.0, rec=True,
                 r=[0.25, 0.05, 0.15], t0=0.0, dv = 0, dt=1e-3, gamma=1.5):
        self.m, self.c, self.k, self.x, self.v, self.r, self.t0, self.dv, self.dt, self.F, self.gamma, self.rec \
            = m, c, k, x, v, r, t0, dv, dt, F, gamma, rec
        self.contact = 0
        self.up = 0
        if rec:
            self.info = {}
            self.info['t'] = [self.t0]
            self.info['x'] = [self.x]
            self.info['v'] = [self.v]
            a = self.func(self.x + self.v, self.t0)
            self.info['a'] = [a[3:6]]
            self.info['c'] = [self.c]
            self.info['contact'] = [self.contact]

    def set(self,c2 = 100.0, k2 = 0.0, F = 0.0, dt = 1e-3):
        self.c[1] = c2
        self.k[1] = k2
        self.F = F
        self.dt = dt
        print('c:', self.c,'\t|','k:', self.k,'\t|','F:', self.F,'\t|','dt:', self.dt,'\t|','t:', self.t0)

    def gen(self, c2 = True, rec = True):
        self.rec = rec
        self.dv = 0
        top_m, low_m = 2000.0, 500.0
        top_c1, low_c1 = 100.0, 10.0
        top_c2, low_c2 = 1000.0, 100.0
        top_k1, low_k1 = 1e12, 1e10
        top_v, low_v = 10, 0.5
        self.m[0] = np.random.random_sample() * (top_m - low_m) + low_m
        self.r[0] = self.r[2] * ((self.m[0] / self.m[2]) ** (1 / 3))
        self.x = [0.0, self.r[0]+ self.r[1], 1000.0]
        self.v = [np.random.random_sample() * (top_v - low_v) + low_v, 0.0, 0.0]
        print('New environment generated!')
        print('m:', self.m, '\t|', 'r:', self.r, '\t|', 'x:', self.x, '\t|', 'v:', self.v, '\t|', 'rec:', self.rec, )
        self.c[0] = np.random.random_sample() * (top_c1 - low_c1) + low_c1
        self.k[0] = np.random.random_sample() * (top_k1 - low_k1) + low_k1
        if c2:
            self.c[1] = np.random.random_sample() * (top_c2 - low_c2) + low_c2
        self.set(c2=self.c[1])
        self.t0 = 0
        self.contact = 0
        if rec:
            self.info = {}
            self.info['t'] = [self.t0]
            self.info['x'] = [self.x]
            self.info['v'] = [self.v]
            a = self.func(self.x + self.v, self.t0)
            self.info['a'] = [a[3:6]]
            self.info['c'] = [self.c]
            self.info['contact'] = [self.contact]

    def func(self, xv, t):
        m, c, k, r, gamma, F = self.m, self.c, self.k, self.r, self.gamma, self.F
        x = xv[0:3]
        v = xv[3:6]
        if self.r[0] + self.r[1] > x[1] - x[0]:
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
        tspan = np.linspace(self.t0,t,int(self.dt * 1e6 + 1))
        sol = odeint(self.func,self.x+self.v,tspan)
        xv = sol[-1].tolist()
        if self.info['v'][-1][0] != xv[3] and self.up == 0:
            self.contact += 1
            self.up = 2
        if self.up > 0:
            self.up -= 1
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
        self.dv += abs(self.info['v'][-1][0] - self.info['v'][-1][1])
        delay = 200
        if self.info['v'].__len__() > delay:
            self.dv -= abs(self.info['v'][-delay][0] - self.info['v'][-delay][1])
            return self.dv > delay/100
        return True


if __name__ == '__main__':
    sample_num = 1
    for i in range(sample_num):
        print(i + 1,'/',sample_num)
        Env = envModel()
        Env.gen()
        while True:
            if not Env.solve():
                break
        x = np.array(Env.info['x'])
        v = np.array(Env.info['v'])
        t = np.array(Env.info['t'])
        r = np.array(Env.r)
        contact = np.array(Env.info['contact'])

        figs = 3

        plt.figure(figs*i)
        plt.plot(t, v[:, 0], 'r', label='v1')
        plt.plot(t, v[:, 1], 'g', label='v2')
        plt.plot(t, v[:, 2], 'b', label='v3')
        plt.xlabel('t')
        plt.grid()
        plt.legend()

        plt.figure(figs*i+1)
        plt.plot(t,x[:,1]-x[:,0]-r[0]-r[1], 'b', label='x2-x1-r2-r1')
        plt.xlabel('t')
        plt.grid()
        plt.legend()

        plt.figure(figs*i+2)
        plt.plot(t,contact, 'r', label='contact')
        plt.xlabel('t')
        plt.grid()
        plt.legend()
    plt.show()