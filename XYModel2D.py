# import labraries
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from itertools import combinations
from tqdm import tqdm
import sys

def pbc_distance(x1, x2, L):
    '''
    Compute distance between two points with periodic boundary condition
    Input: 
        x1,x2: 2d position of one point
        L: size of the system
    Output:
        distance under the condition
    '''
    dx = abs(x1[0] - x2[0])
    dx = min(dx, L - dx)
    
    dy = abs(x1[1] - x2[1])
    dy = min(dy, L - dy)
    return np.sqrt(dx**2 + dy**2)

class XYModel2D:
    '''
    This class can create and simulate a 2d xy model with given size and temperature
    '''
    def __init__(self, L, T, J=1, T_seperate=1.3):
        '''
        Initialises a random spin configuration with system size L and temperature T
        Args:
            L: size of the system
            T: temperature of the system
            J: coupling strength, default=1
            T_seperate: if T<T_seperate, annealing simulation is activated.
                        otherwise, use normal monte carlo simualtion
                        default=1.3
        '''
        self.L = L
        self.T = T
        self.J = J
        self.T_seperate = T_seperate
        self.t_interval = L * L # number of simulation for '1 second'
        self.theta = np.random.uniform(0, 2 * np.pi, (L, L)) # generate a random spin configuration

    def __delta_energy(self, x, y, new_angle):
        '''
        Compute energy difference by changing the angle of a random spin
        Args:
            x,y: the position of the spin
            new_angle: the new angle
        '''
        L = self.L
        theta = self.theta
        
        dE = 0.0
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            xn, yn = (x + dx) % L, (y + dy) % L # position of the nearest neighbours
            dE += -self.J * (np.cos(theta[xn, yn] - new_angle) -
                             np.cos(theta[xn, yn] - theta[x, y])) # function of energy difference
        return dE

    def monte_carlo_step(self):
        '''
        One step of monte carlo simulation
        '''
        x, y = np.random.randint(0, self.L), np.random.randint(0, self.L) # randomly choose a spin
        new_angle = np.random.uniform(0, 2*np.pi) # change the angle
        dE = self.__delta_energy(x, y, new_angle) # compute the energy difference
        
        if dE < 0 or np.random.rand() < np.exp(-dE / self.T): # accept the change for different case
            self.theta[x, y] = new_angle

    def SimulationFor1Time(self):
        '''
        Do simulation for '1 second'
        '''
        for _ in range(self.t_interval):
            self.monte_carlo_step()

    def total_energy(self):
        '''
        Compute the total energy of the system.
        '''
        E = 0.0
        for i in range(self.L):
            for j in range(self.L):
                theta = self.theta[i, j]
                # left and down neighbours of one spin
                neighbors = [
                    ((i + 1) % self.L, j),
                    (i, (j + 1) % self.L)
                ]
                for ni, nj in neighbors:
                    E -= self.J * np.cos(theta - self.theta[ni, nj])
        return E
    
    def __equilibrate_lowT(self, max_steps=10_000, energy_temp = None):
        '''
        Equilibrate the system for low temperature
        Args:
            max_steps: the maximum steps to reach equilibrium, default=10_000
            energy_temp: energy inherited from the previous temperature simulation, default=None
        '''
        if energy_temp is None: # if energy_temp=None, we are at the first temperature simulation
            energy_temp = np.inf # set to infinity

        for k in range(max_steps):
            self.SimulationFor1Time() # simulate for 1s
            energy = self.total_energy()
            if ((abs(energy-energy_temp)/abs(energy)<1e-5)&(k>500)) or k == max_steps-1: # check if the energy converges
                break
            energy_temp = energy # update the previous energy

    def anneal_simulation(self, T_end, T_step=0.02):
        '''
        Annealing process
        Arg:
            T_end: the target temperature we want to achieve
            T_step: the small drop in temperature for annealing, default=0.05
        '''
        T_start=self.T_seperate
        T_list = np.linspace(T_start, T_end, num=int((T_start - T_end)/T_step) + 1)
        
        print(f"\nStarting annealing simulation at T = {T_end:.3f} ")
        
        self.T=T_start
        self.__equilibrate_lowT()
        
        for T in T_list[1:]:
            self.T = T
            self.__equilibrate_lowT(energy_temp=self.total_energy())
        print('equilibrium state is reached at T=%.3f'%self.T)
    
    def __equilibrate_highT(self, max_steps=10_000):
        '''
        Equilibrate the system for high temperature
        Args:
            max_steps: the maximum steps to reach equilibrium, default=10_000
        '''
        print(f"\nStarting normal simulation at T = {self.T:.3f} ")

        energy_temp = np.inf # set to infinity

        for k in range(max_steps): 
            self.SimulationFor1Time() # simulate for 1s
            energy = self.total_energy()
            if ((abs(energy-energy_temp)/abs(energy)<1e-5) &(k>500)) or k == max_steps-1: # check if the energy converges
                print('equilibrium state is reached at T=%.3f'%self.T)
                break
            energy_temp = energy # update the previous energy
    
    def equilibrate(self):
        '''
        The integrated equilibrium process
        '''
        if self.T<self.T_seperate: # if T<T_seperate, start annealing
            self.anneal_simulation(self.T)
        else: # otherwise, start normal simulation process
            self.__equilibrate_highT()

    def compute_vorticity(self):
        '''
        Compute the vorticity for a given spin configuration
        '''
        L = self.L
        vort = [] # set of vorticity

        for i in range(L):
            for j in range(L): # go through all combinations
                angles = [
                    self.theta[i, j],
                    self.theta[i, (j+1)%L],
                    self.theta[(i+1)%L, (j+1)%L],
                    self.theta[(i+1)%L, j]
                ] # 2x2 spins

                delta = []
                for k in range(4):
                    d = angles[(k+1)%4] - angles[k]
                    d = (d + np.pi) % (2 * np.pi) - np.pi
                    delta.append(d) # difference between spins
                winding = sum(delta) # winding number
                n = int(np.round(winding / (2 * np.pi)))
                if n != 0: # collect non-zero values and their positions
                    vort.append([i,j,n])
        return vort
    
    def vortex_vortex_interaction(self, a=1):
        '''
        Compute vortex-vortex interactions
        Arg:
            a: the space between two spins, default=1
        '''

        vort = self.compute_vorticity() # find the vortices
        mu_v = 3.56 * (self.J - self.T / 4) # effective core energy
        if len(vort) < 2: # since vortices are paired, at least two appear at a time
            return 0
        V = 0
        for v_pair in combinations(vort, 2): 
            V += v_pair[0][-1]* v_pair[1][-1]*np.log(pbc_distance(v_pair[0][:2], v_pair[1][:2], self.L)/a)
        return -np.pi * self.J * V + mu_v*len(vort) # interaction equation

    def plot_spin_with_vortices(self):
        '''
        Visualisation for spin configuration
        '''
        L = self.L
        U = np.cos(self.theta)
        V = np.sin(self.theta)
        vort = self.compute_vorticity()
    
        fig, ax = plt.subplots(figsize=(6, 6))
        X, Y = np.meshgrid(np.arange(L), np.arange(L))
        # Plot spin field
        ax.quiver(X, Y, U, V, pivot='mid', scale=50)
        ax.set_xticks([]); ax.set_yticks([])
        # Overlay vortex markers at plaquette centers
        # switch x and y position because of the differnt logic for plotting
        for v in vort:
            if v[-1] == 1:
                ax.plot(v[1]+0.5, v[0]+0.5, 'ro', markersize=6, label='Vortex')
            elif v[-1] == -1:
                ax.plot(v[1]+0.5, v[0]+0.5, 'bo', markersize=6, label='Anti-Vortex')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

    def sample(self, n_samples):
        '''
        Compute the samples of macro and micro.
        Arg:
            n_samples: number of samples required
        '''
        V_t = [] # macro
        X_t = [] # micro
        for i in range(n_samples):
            self.SimulationFor1Time()
            V_t.append(self.vortex_vortex_interaction())
            X_t.append(self.theta.flatten())
        return V_t, X_t
    
    def heat_capacity(self, n_samples=1000):
        '''
        Compute the heat capacity of the system with given L and T
        '''
        energies = [] # total energy
        energies_squared = [] # square of total energy
        for _ in range(n_samples):
            self.SimulationFor1Time()
            E = self.total_energy()
            energies.append(E)
            energies_squared.append(E ** 2)
        # take the average
        E_avg = np.mean(energies)
        E2_avg = np.mean(energies_squared)
        return (E2_avg - E_avg ** 2) / (self.L * self.L * self.T**2 )
    
    def helicity_modulus_x(self, n_samples=1000):
        """
        compute helicity modulus in x direction
        """
        beta = 1.0 / self.T
        L = self.L
        N = L * L

        H = 0.0 # total energy
        sum_sin_sq = 0.0 # square of sum of sin
        
        for _ in range(n_samples):
            self.SimulationFor1Time() 

            # compute coupling strength on the right
            S_x = 0.0
            for i in range(L):
                for j in range(L):
                    right_j = (j + 1) % self.L
                    dtheta = self.theta[i, j] - self.theta[i, right_j]
                    S_x += np.sin(dtheta)

            sum_sin_sq += S_x ** 2
            H += self.total_energy()

        H_avg = H / n_samples
        sin_sq_avg = sum_sin_sq / n_samples

        term1 = - H_avg / (2 * N)
        term2 = - (self.J / (self.T * N)) * sin_sq_avg
        return term1 + term2
    
