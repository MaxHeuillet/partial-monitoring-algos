#!/usr/bin/env python
# coding: utf-8
# """Partial Monitoring Library:
# Provides a collection of Finite Partial Monitoring algorithms for experimental studies.
# """
# __author__ = "Tanguy Urvoy"
# __copyright__ = "Orange-labs, France"
# __license__ = "GPL"
# __version__ = "1.2"
# __email__ = "tanguy.urvoy@orange.com"
# __date__ = "2017"
# __status__ = "Beta"

import numpy as np

## Convex polyhedron manipulation library
import ppl #parma polyhedra library for the cell decomposition
from itertools import islice
import gurobipy as gp
from gurobipy import GRB


def get_P_t(halfspace, L):
    P_t  = []
    N, M = L.shape
    for i in range(N):
        result = single_cell_intersection(i, L, halfspace)
        if result.is_empty() == False:
            P_t.append(i)
    return P_t

def get_N_t(halfspace, L):
    N_t  = []
    N, M = L.shape
    for i in range(N):
        for j in range(N):
            if two_cell_intersection([i,j], L, halfspace ):
                N_t.append([i,j])
    return N_t



def two_cell_intersection(pair, LossMatrix, halfspace):

    N, M = LossMatrix.shape
    # declare M ppl Variables

    cs = ppl.Constraint_System()
    p = [ ppl.Variable(j) for j in range(M) ]
    cs.insert( sum( p[j] for j in range(M)) == 1 )
    for j in range(M):
        cs.insert(p[j] >= 0)

    # intersection from the halfspaces:
    for element in halfspace:
        pair, sign = element[0], element[1]
        Lij = LossMatrix[ pair[0] ] - LossMatrix[ pair[1] ]  
        cs.insert(  sign * sum( ( Lij[a] * p[a] for a in range(M) ) )  > 0 )

    open_polytope = ppl.NNC_Polyhedron(cs) 
    # print('open polytope', open_polytope)

    cs2 = ppl.Constraint_System()
    p2 = [ ppl.Variable(j) for j in range(M) ]
    cs2.insert( sum( p2[j] for j in range(M)) == 1 )
    for j in range(M):
        cs.insert(p2[j] >= 0)

    # strict Loss domination constraints for both a and b
    for i in range(N):
        if i!=pair[0]:
            # p is such that for any action i Loss[a,...]*p <= Loss[a,...]*p
            cs2.insert( LossMatrix[ pair[0],...].dot(p) - LossMatrix[i,...].dot(p2) <= 0 )
        if i!=pair[1]:
            # p is such that for any action i Loss[b,...]*p <= Loss[a,...]*p
            cs2.insert( LossMatrix[ pair[1],...].dot(p) - LossMatrix[i,...].dot(p2) <= 0 )

    Ci_inter_Cj = ppl.NNC_Polyhedron(cs2) 
    # print('Ci_inter_Cj', Ci_inter_Cj)

    return intersects(open_polytope, Ci_inter_Cj)

def intersects(A, B):
    if A.is_empty() or B.is_empty():
        return True
    else:
        A.intersection_assign(B)
        # print('intersection:, ', A)
        return A.is_empty()



def single_cell_intersection(i, LossMatrix, halfspace):

    N, M = LossMatrix.shape
    
    p = [ ppl.Variable(j) for j in range(M) ] # declare M ppl Variables
    cs = ppl.Constraint_System() # declare polytope constraints

    # p belongs to $\Delta_M$ the set of M dimensional probability vectors
    cs.insert( sum( p[j] for j in range(M)) == 1 )
    for j in range(M):
        cs.insert(p[j] >= 0)

    # strict Loss domination constraints
    # Dom = scale_to_integers(domination_matrix(i,LossMatrix))
    
    for a in range(N):
        if a != i:
            # p is such that for any action a Loss[i,...]*p <= Loss[a,...]*p
            #print "Domination line:", Dom[a,...], "inequality:", sum( (Dom[a,j]*p[j] for j in range(M)) ) <= 0
            cs.insert( LossMatrix[i,...].dot(p) <= 0 )
    
    # intersection from the halfspaces:
    for element in halfspace:
        pair, sign = element[0], element[1]
        Lij = LossMatrix[ pair[0] ] - LossMatrix[ pair[1] ]  
        cs.insert(  sign * sum( ( Lij[a] * p[a] for a in range(M) ) )  > 0 )

    return ppl.NNC_Polyhedron(cs)


### General FeedExp3 helpers

# build the signal vector i.e. [F_{i,j}=v]_{j=1,...,M}
# (with NxM format matrices)

def signal_vec(i, v, FeedbackMatrix):
    return (FeedbackMatrix[i,...] == v).astype(np.float)

# build the signal matrix  i.e. [F_{i,j}=v]_{j=1,...,M}

def signal_vecs(i, FeedbackMatrix):
    return [signal_vec(i, v, FeedbackMatrix) for v in set(FeedbackMatrix[i,...])]


# Fjv: the row binary signal matrix for F_{i,j}=v (pseudo-action i/v)
# Fdash_list : a list of row feedback vectors
# check if a signal matrix is in the linear combination of the previous feedback vectors
# i.e. that staking this new vector does not increase matrix rank

def is_linear_comb(Fiv, Fdash_list):
    if len(Fdash_list) == 0:
        return True
    initial_rank = np.linalg.matrix_rank(np.vstack(Fdash_list))
    new_rank = np.linalg.matrix_rank(np.vstack(Fdash_list + [Fiv]))
    return new_rank == initial_rank


## Domination Cells decomposition of a Partial Monitoring Game

# Domination matrix is the upper bound constraint for the i-th action's PM Cell
# (LossMatrix[i,...] - LossMatrix).dot(p) < 0 means that 
# i is the best action for outcome distribution p
def domination_matrix(i,LossMatrix):
    return LossMatrix[i,...] - LossMatrix

# transform a floating point Domination matrix into an equivalent integer matrix
def scale_to_integers(Dom):
    where = np.modf(Dom)[0] != 0
    if where.any():
        #print "WARNING: ppl works only with integers and silently removes frational part of floats"
        #print "Rescaling Domination Matrix!"
        m = np.min(Dom[where])
        return Dom/np.abs(m)
    return Dom

# return domination Cell polytope for action i
def DominationPolytope(i,LossMatrix):
    N, M = LossMatrix.shape

    # declare M ppl Variables
    p = [ppl.Variable(j) for j in range(M)]
    
    # declare polytope constraints
    cs = ppl.Constraint_System()
    
    # probabilies constraints on p
    cs.insert( sum( p[j] for j in range(M)) == 1 )
    for j in range(M):
        cs.insert(p[j] >= 0)
        
    # strict Loss domination constraints
    Dom = scale_to_integers(domination_matrix(i,LossMatrix))
    
    for a in range(N):
        if a != i:
            # p is such that for any action a Loss[i,...]*p <= Loss[a,...]*p
            #print "Domination line:", Dom[a,...], "inequality:", sum( (Dom[a,j]*p[j] for j in range(M)) ) <= 0
            cs.insert( sum( (Dom[a,j]*p[j] for j in range(M)) ) <= 0 )
            
    return ppl.C_Polyhedron(cs)

def get_observer_vector_v2(pair, L, H ):
    
    # Lij = L[pair[0],...] - L[pair[1],...]
    
    # S_vectors = [ signal_vecs(k, H) for k in [0,1] ]
    # stacked_S =  np.linalg.pinv(  np.vstack( S_vectors ).T )

    # resultat = stacked_S * Lij 
    # v_ij = resultat.T[0]
    # print(v_ij)
    # length = [ len(k)  for k in S_vectors]
    # v_ij = iter( v_ij )
    # return [ np.array( list( islice( v_ij, i)) ) for i in length] 

    if pair == [0,0] or pair == [1,1]:
        return [ np.array([0]), np.array([0,0]) ]
    if pair == [0,1]:
        return [ np.array([1/3]),np.array([-1,1/3]) ]
    if pair == [1,0]:
        return [ np.array([1/3]),np.array([1/3,1]) ]

def get_neighborhood_action_set(pair, N_bar, L):
    
    cell_i = DominationPolytope(pair[0], L)
    cell_j = DominationPolytope(pair[1], L)
    cell_i.intersection_assign(cell_j) 
    mathcal_N_plus = []
    for k in N_bar:
        cell_k = DominationPolytope(k, L)
        # print( cell_k.contains(cell_i) )
        if cell_k.contains(cell_i):
            mathcal_N_plus.append(k)
    return mathcal_N_plus

# return domination Cell polytope interior for action i
def StrictDominationPolytope(i,LossMatrix):
    N, M = LossMatrix.shape

    # declare M ppl Variables
    p = [ppl.Variable(j) for j in range(M)]
    
    # declare polytope constraints
    cs = ppl.Constraint_System()
    
    # probabilies constraints on p
    cs.insert( sum( p[j] for j in range(M)) == 1 )
    for j in range(M):
        cs.insert(p[j] >= 0)
        
    # strict Loss domination constraints
    Dom = scale_to_integers(domination_matrix(i,LossMatrix))    

    for a in range(N):
        if (Dom[a,...] != 0).any():
            # p is such that for any action a Loss[i,...]*p <= Loss[a,...]*p
            #print "Strict Domination line:", Dom[a,...], "inequality:", sum( (Dom[a,j]*p[j] for j in range(M)) ) < 0
            cs.insert( sum( (Dom[a,j]*p[j] for j in range(M)) ) < 0 )
            
    return ppl.NNC_Polyhedron(cs)



# Check that an action is dominant
# Check that an action is strictly dominant 
# i.e. there exists some outcome distributions where i is one of the best actions
def isNonDominated(i, LossMatrix):
    return not (DominationPolytope(i,LossMatrix).is_empty())

# Check that an action is strictly dominant 
# i.e. there exists some outcome distributions where i is strictly the best action
def isStrictlyNonDominated(i, LossMatrix):
    return not (StrictDominationPolytope(i,LossMatrix).is_empty())

# Check if an action is degenerated
# i.e. if there exists another action Cell containing strictly its cell.
def isDegenerated(i,LossMatrix):
    N,M = LossMatrix.shape
    polytope_i = DominationPolytope(i, LossMatrix)
    if polytope_i.is_empty():
        return False
    isDegen = False
    j=0
    while(not isDegen and j<N):
        if j!=i:
            # strict inclusion test
            if polytope_i < DominationPolytope(j, LossMatrix):
                #print "Cell(",i,") is strictly inside Cell(", j, ")"
                isDegen = True
        j += 1
    return isDegen       

# Check if an action is pareto optimal
def isParetoOptimal(i, LossMatrix):
    return isNonDominated(i, LossMatrix) and not isDegenerated(i,LossMatrix)


# Return the polytope where both a and b are winning actions
def interFacePolytope(a, b, LossMatrix):
    N, M = LossMatrix.shape

    # declare M ppl Variables
    p = [ppl.Variable(j) for j in range(M)]
    
    # declare polytope constraints
    cs = ppl.Constraint_System()
    
    # probabilies constraints on p
    cs.insert( sum( p[j] for j in range(M)) == 1 )
    for j in range(M):
        cs.insert(p[j] >= 0)
        
    # strict Loss domination constraints for both a and b
    Doma = scale_to_integers(domination_matrix(a,LossMatrix))
    Domb = scale_to_integers(domination_matrix(b,LossMatrix))        
    for i in range(N):
        if i!=a:
            # p is such that for any action i Loss[a,...]*p <= Loss[a,...]*p
            cs.insert( sum( (Doma[i,j]*p[j] for j in range(M)) ) <= 0 )
        if i!=b:
            # p is such that for any action i Loss[b,...]*p <= Loss[a,...]*p
            cs.insert( sum( (Domb[i,j]*p[j] for j in range(M)) ) <= 0 )
            
    return ppl.C_Polyhedron(cs)

# Check if two actions are neighbours
def areNeighbours(a, b, LossMatrix):
    M = LossMatrix.shape[1]
    return interFacePolytope(a, b, LossMatrix).affine_dimension() >= M - 2

# full signal space
def global_signal(FeedbackMatrix):
    return [signal_vecs(i, FeedbackMatrix) for i in range(FeedbackMatrix.shape[0])]

# Returns the neigbourhood of a pair of actions
def Neighbourhood(a, b, LossMatrix):
    N, M = LossMatrix.shape
    
    frontier = interFacePolytope(a, b, LossMatrix)
    
    Nb = []
    for k in range(N):
        if k==a or k==b or frontier <= DominationPolytope(k, LossMatrix):
            Nb.append(k)
    
    return Nb

def is_linear_comb(Fiv, Fdash_list):
    if len(Fdash_list) == 0:
        return True
    initial_rank = np.linalg.matrix_rank(np.vstack(Fdash_list))
    new_rank = np.linalg.matrix_rank(np.vstack(Fdash_list + [Fiv]))
    return new_rank == initial_rank


# Observability for a pair of actions
def ObservablePair(a, b, LossMatrix, S_list):
    Lab = LossMatrix[a,...] - LossMatrix[b,...]
    return is_linear_comb(Lab, S_list)

# Global observability for a game
def GlobalObservableGame(pm):
    LossMatrix, FeedbackMatrix = pm.LossMatrix, pm.FeedbackMatrix
    N,M = pm.N, pm.M
    assert (N,M) == LossMatrix.shape
    assert (N,M) == FeedbackMatrix.shape
    
    global_S_list = global_signal(FeedbackMatrix)

    res = True
    why = "all pairs are globally observable."
    for a in range(N):
        for b in range(a+1,N):
            res = res and ObservablePair(a, b, LossMatrix, global_S_list)
            if not res:
                why = "[{0},{1}] pair is not globally observable.".format(pm.Actions_dict[a],pm.Actions_dict[b])
                return res, why
    return res, why


# Global observability for a game
def LocalObservableGame(pm):
    LossMatrix, FeedbackMatrix = pm.LossMatrix, pm.FeedbackMatrix
    N,M = pm.N, pm.M
    assert (N,M) == LossMatrix.shape
    assert (N,M) == FeedbackMatrix.shape

    why = "all neighbouring pairs are observable."
    res = True
    for a in range(N):
        for b in range(a+1,N):
            if areNeighbours(a, b, LossMatrix):
                # local signal space
                local_S_list = [signal_vecs(i, pm.FeedbackMatrix) for i in Neighbourhood(a, b, LossMatrix)]
                
                res = res and ObservablePair(a, b, LossMatrix, local_S_list)
            if not res:
                why = "[{0},{1}] pair is not locally observable.".format(pm.Actions_dict[a], pm.Actions_dict[b])
                return res, why
    return res, why


def isTrivialGame(pm):
    ndom = [isNonDominated(i, pm.LossMatrix) for i in range(pm.N)]
    K = sum(ndom)
    if K==1:
        first = range(pm.N)[ndom]
        return True, "action {0} is always the best.".format(pm.Actions_dict[first[0]])
    else:
        return False, "there are more than one dominant actions"
    

def ProblemClass(pm):
    trivial, why = isTrivialGame(pm)
    if trivial:
        return "trivial", why
    else:
        glob, why = GlobalObservableGame(pm)
        if glob:
            loc, why = LocalObservableGame(pm)
            if loc:
                return "easy", why
            else:
                return "hard", why
        else:
            return "intractable", why
