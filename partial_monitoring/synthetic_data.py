
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm


class QuinticContexts:
    def __init__(self, d, margin):
        self.d = d #number of features
        self.margin = margin #np.random.uniform(0,0.5) # decision boundary

    def get_context(self, label):
        while True:
            context = np.random.uniform(-1, 1,  self.d )
            x,y = context
            if   y < -0.25+x**2+0.1*x**3+0.05*x**4+0.01*x**5-0.4*x**6 - self.margin  and label == 0:
                return np.array(context).reshape(2,1) #/ norm(context, 1)
            elif   y >= -0.25+x**2+0.1*x**3+0.05*x**4+0.01*x**5-0.4*x**6 + self.margin and label == 1:
                return np.array(context).reshape(2,1) #/ norm(context, 1)

    def generate_unique_context(self,):
        self.context_A = []
        self.context_B = []
        while  len(self.context_A) == 0 or len(self.context_B) == 0:
            context = np.random.uniform(-1, 1, self.d)
            x,y = context
            if y < -0.25+x**2+0.1*x**3+0.05*x**4+0.01*x**5-0.4*x**6 + self.margin and len(self.context_A) == 0 :
                self.context_A = np.array(context).reshape(2,1) #/ norm(context, 1)
            elif y >= -0.25+x**2+0.1*x**3+0.05*x**4+0.01*x**5-0.4*x**6 + self.margin and len(self.context_B) == 0:
                self.context_B = np.array(context).reshape(2,1) #/ norm(context, 1)


    def get_same_context(self, label):
        if label == 0:
                return self.context_A
        elif label == 1:
                return self.context_B

class LinearContexts:
    def __init__(self, w,b, d, margin):
        self.d = d #number of features
        self.margin = margin #np.random.uniform(0,0.5) # decision boundary
        self.b = b #np.random.uniform(-1,1)
        self.w = w #np.random.uniform(-1,1, self.d) 

    def get_context(self, label):
        while True:
            context = np.random.uniform(-1, 1,  self.d )
            if   self.w.T @ context + self.b > self.margin and label == 0:
                return np.array(context).reshape(2,1) #/ norm(context, 1)
            elif  self.w.T @ context  + self.b < -self.margin and label == 1:
                return np.array(context).reshape(2,1) #/ norm(context, 1)

    def generate_unique_context(self,):
        self.context_A = []
        self.context_B = []
        while  len(self.context_A) == 0 or len(self.context_B) == 0:
            context = np.random.uniform(-1, 1, self.d)
            if self.w.T @ context + self.b > self.margin and len(self.context_A) == 0 :
                self.context_A = np.array(context).reshape(2,1) #/ norm(context, 1)
            elif self.w.T @ context + self.b < -self.margin and len(self.context_B) == 0:
                self.context_B = np.array(context).reshape(2,1) #/ norm(context, 1)


    def get_same_context(self, label):
        if label == 0:
                return self.context_A
        elif label == 1:
                return self.context_B

class ToyContexts:

    def __init__(self, ):
        pass

    def get_context(self, label):
        while True:
            context =   np.random.uniform(0, 1 ) # np.random.randint(2)
            if   context >= 0.5 and label == 0:
                return np.array([1,context]).reshape(2,1) #/ norm(context, 1)
            elif  context < 0.5 and label == 1:
                return np.array([1,context]).reshape(2,1) #/ norm(context, 1)

    def generate_unique_context(self,):
        self.context_A = []
        self.context_B = []
        while  len(self.context_A) == 0 or len(self.context_B) == 0:
            context =  np.random.uniform(0, 1) #np.random.randint(2)
            if context >= 0.5 and len(self.context_A) == 0 :
                self.context_A = np.array([1,context]).reshape(2,1) #/ norm(context, 1)
            elif context < 0.5 and len(self.context_B) == 0:
                self.context_B = np.array([1,context]).reshape(2,1) #/ norm(context, 1)

    def get_same_context(self, label):
        if label == 0:
                return self.context_A
        elif label == 1:
                return self.context_B