
import numpy as np
import matplotlib.pyplot as plt

class LinearContexts:
    def __init__(self, d, margin):
        self.d = d
        self.margin = np.random.uniform(0,0.5)

    def make_task(self,):
        control = False
        while control == False: # ensure that the task will allow both contexts to exist
            self.b = np.random.uniform(-1,1)
            self.w = np.random.uniform(-1,1, self.d) 
            classA, classB = [], []
            for _ in range(1000):
                point = np.random.uniform(-1, 1, self.d)
                if self.w.T @ point + self.b > self.margin:
                    classA.append(point)
                elif self.w.T @ point + self.b < -self.margin:
                    classB.append(point)
            if len(classA)>0 and len(classB)>0:
                control = True

    def get_context(self, label):
        while True:
            context = np.random.uniform(-1, 1, self.d)
            if self.w.T @ context + self.b > self.margin and label == 0:
                return context
            elif self.w.T @ context + self.b < -self.margin and label == 1:
                return context

    def generate_unique_context(self,):
        self.context_A = []
        self.context_B = []

        while  len(self.context_A) == 0 or len(self.context_B) == 0:

            context = np.random.uniform(-1, 1, self.d)
            if self.w.T @ context + self.b > self.margin and len(self.context_A) == 0 :
                self.context_A = context
            elif self.w.T @ context + self.b < -self.margin and len(self.context_B) == 0:
                self.context_B = context


    def get_same_context(self, label):
        context = np.random.uniform(-1, 1, self.d)
        if label == 0:
                return self.context_A
        elif label == 1:
                return self.context_B