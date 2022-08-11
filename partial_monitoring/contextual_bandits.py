import numpy

class ContextualLinGaussianPM:
    
    def __init__(self, thetas, noise, seed=None):
        self.thetas = numpy.copy(thetas)
        self.sigma = noise
        self.random = numpy.random.RandomState(seed)
        
        self.K = thetas.shape[0]
        self.d = thetas.shape[1]
        
        self.context = None
        
        self.regret = []
    
    def get_K(self):
        return self.K
    
    def get_context(self):
        self.context = self.random.uniform(-1, 1)
        return self.context
    
    def play(self, k):
        phi_s = numpy.array([1, self.context, self.context**2, self.context**3, self.context**4])
        means = self.thetas.dot(phi_s)
        k_star = numpy.argmax(means)
        self.regret.append(means[k_star] - means[k])
        return means[k] + self.random.normal(0, self.sigma)
    
    def get_cumulative_regret(self):
        return numpy.cumsum(self.regret)