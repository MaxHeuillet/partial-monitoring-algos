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





### General FeedExp3 helpers

# build the signal vector i.e. [F_{i,j}=v]_{j=1,...,M}
# (with NxM format matrices)

def signal_vec(i, v, FeedbackMatrix):
    return (FeedbackMatrix[i,...] == v).astype(np.float)

# build the signal matrix  i.e. [F_{i,j}=v]_{j=1,...,M}

def signal_vecs(i, FeedbackMatrix):
    return [signal_vec(i, v, FeedbackMatrix) for v in set(FeedbackMatrix[i,...])]


def get_alphabet_size(FeedbackMatrix):
    N, M = FeedbackMatrix.shape
    letters = []
    for i in range(N):
        for j in range(M):
            letters.append( FeedbackMatrix[i][j] )
    return len(set(letters)) 


def get_signal_matrices(H):
    N, M = H.shape
    A = get_alphabet_size(H)
    signal_matrices = []
    for i in range(N):
        signal_matrix = np.zeros( (A,M) )
        for j in range(M):
            a = H[i][j]
            signal_matrix[a][j] = 1
        signal_matrices.append(signal_matrix)
    return signal_matrices



    # # print(S_vectors)
    # stacked_S =  np.linalg.pinv(  np.vstack( S_vectors ).T )

    # resultat = stacked_S * Lij 
    # v_ij = resultat.T[0]
    # length = [ len(k)  for k in S_vectors]
    # v_ij = iter( v_ij )
    # return [ np.array( list( islice( v_ij, i)) ) for i in length] 



# def get_observer_vector(pair, L,H, observer_set):
    
#     Lij = L[pair[0],...] - L[pair[1],...]
#     S_vectors = [ signal_vecs(k, H) for k in observer_set[ pair[0] ][ pair[1] ] ]
#     # print(S_vectors)
#     stacked_S =  np.linalg.pinv(  np.vstack( S_vectors ).T )

#     resultat = stacked_S * Lij 
#     v_ij = resultat.T[0]
#     length = [ len(k)  for k in S_vectors]
#     v_ij = iter( v_ij )
#     return [ np.array( list( islice( v_ij, i)) ) for i in length] 



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




def get_observer_vector(pair, L,H, S_vectors):

    N, M = H.shape
    Lij = L[pair[0],...] - L[pair[1],...]
    A = get_alphabet_size(H)
    

    m = gp.Model("mip1")
    m.Params.LogToConsole = 0

    vars = []
    for k in range(N):
        vars.append( [] )
        for a in range(A):
            varName =  '{} {}'.format(k,a) 
            vars[k].append( m.addVar(-GRB.INFINITY, GRB.INFINITY, 0., GRB.CONTINUOUS, varName) )

    m.update()

    obj = gp.QuadExpr ()
    for i in range(N):
        for a in range(A):
            obj += vars[k][a] * vars[k][a]

    m.setObjective(obj, GRB.MINIMIZE)

    for j in range(M):
        constraintExpr = gp.LinExpr()
        str = 'c {}'.format(j)
        for a in range(A):
            for k in range(N):
                constraintExpr += S_vectors[k][a][j] * vars[k][a]
        m.addConstr(constraintExpr == Lij[j], str );

    m.optimize()
    vij = np.zeros( (N, A) )
    for k in range(N):
        for a in range(A):
            vij[k, a] = vars[k][a].X

    return vij


def isNeighbor(LossMatrix, i1, i2, halfspace):

    feasible = True
    N, M = LossMatrix.shape
    #print('N', N, 'M', M, 'i1', i1, 'i2', i2)

    m = gp.Model("mip1")
    m.Params.LogToConsole = 0

    vars = []
    for j in range(M):
        varName =  'p_{}'.format(j) 
        vars.append( m.addVar(0.00001, 1.0, -1.0, GRB.CONTINUOUS, varName) )

    m.update()

    simplexExpr = gp.LinExpr()
    for j in range(M):
        simplexExpr += 1.0 * vars[j]
    m.addConstr(simplexExpr == 1.0, "css")

    twoDegenerateExpr = gp.LinExpr()
    for j in range(M):
        twoDegenerateExpr += ( LossMatrix[i2][j]-LossMatrix[i1][j] ) * vars[j]
    m.addConstr(twoDegenerateExpr == 0.0, "cdeg");

    for i3 in range(N):
        if ( (i3 == i1) or (i2 == i1) ) :
            pass 
        else:
            lossExpr  = gp.LinExpr()
            for j in range(M):
                lossExpr += ( LossMatrix[i3][j] - LossMatrix[i1][j] ) * vars[j]
        
            lossConstStr = "c{}".format(i3)
            m.addConstr(lossExpr >= 0.0, lossConstStr)

    for element in halfspace:
        pair, sign = element[0], element[1]
        if sign == 0:
            pass
        else:
            halfspaceExpr = gp.LinExpr()
            for j in range(M):
                halfspaceExpr += ( sign * (LossMatrix[i1][j] - LossMatrix[i2][j] ) ) * vars[j]
        
            halfspaceConstStr = "ch_{}_{}".format(i1,i2)
            m.addConstr(halfspaceExpr >= 0.0000000000001,  halfspaceConstStr )

    m.optimize()

    if m.getAttr( GRB.Attr.Status )  in ( GRB.INF_OR_UNBD , GRB.INFEASIBLE , GRB.UNBOUNDED ):
        feasible = False

    return feasible
    


    # #simple constraint:
    # p = [ ppl.Variable(j) for j in range(M) ] # declare M ppl Variables
    # cs = ppl.Constraint_System() # declare polytope constraints

    # # p belongs to $\Delta_M$ the set of M dimensional probability vectors
    # cs.insert( sum( p[j] for j in range(M)) == 1 )
    # for j in range(M):
    #     cs.insert(p[j] >= 0)

    # # < p, loss(i1) - loss(i2) > = 0 
    # result = 0
    # for j in range(M):
    #     result += ( LossMatrix[i2][j] - LossMatrix[i1][j] ) * p[j]
    # cs.insert( result == 0)

    # # < p, loss(i1) - loss(i2) > = 0 
    # loss = []
    # for i3 in range(N):
    #     #print('i3',i3,'i2',i2,'i1',i1)
    #     if i3 == i1 or i3 == i2:
    #         pass
    #     else:
    #         for j in range(M):
    #             loss.append(  (  LossMatrix[i3][j] - LossMatrix[i1][j]) * p[j] )
    # #print('loss', loss, sum(loss), len(loss))
    # if len(loss) > 0:
    #     cs.insert( sum(loss) >= 0)

    # # halfspace constraint: h(i,j) * (loss[i]-loss[j])^top @ p > 0 
    # halfspaceExpr = []
    # for element in halfspace:
    #     pair, sign = element[0], element[1]
    #     if sign== 0:
    #         pass
    #     else:
    #         for j in range(M):
    #             coef = sign * ( LossMatrix[ pair[0] ][j] - LossMatrix[  pair[1] ][j] )
    #             if(coef != 0):
    #                 halfspaceExpr.append( coef * p[j] )
        
    # #print('halfspaceexpr', halfspaceExpr)
    # if len(halfspaceExpr) > 0:
    #     cs.insert( sum(halfspaceExpr) > 0 )

    # return ppl.NNC_Polyhedron(cs)


def getNeighbors(LossMatrix, mathcal_N, halfspace):
    N_t = []
    for pair in mathcal_N:
        if isNeighbor(LossMatrix, pair[0], pair[1], halfspace):
            N_t.append( pair )
    return N_t


def getParetoOptimalActions(LossMatrix, halfspace):

    N, M = LossMatrix.shape
    P = []

    for i in range(N):

        p = [ ppl.Variable(j) for j in range(M) ] # declare M ppl Variables
        cs = ppl.Constraint_System() # declare polytope constraints

        # p belongs to $\Delta_M$ the set of M dimensional probability vectors
        cs.insert( sum( p[j] for j in range(M)) == 1 )
        for j in range(M):
            cs.insert(p[j] >= 0)

        # <p , li2 - li > \geq 0
        loss = []
        for i2 in range(N):
            if i2 == i:
                pass
            else:
                for j in range(M):
                    loss.append(  ( LossMatrix[i2][j] - LossMatrix[i][j] ) * p[j] )
        if len(loss)>0:
            cs.insert( sum(loss) >= 0 )

        # halfspace constraint: h(i,j) * (loss[i]-loss[j])^top @ p > 0 
        halfspaceExpr = []
        for element in halfspace:
            pair, sign = element[0], element[1]
            if sign == 0:
                pass
            else:
                for j in range(M):
                    coef = sign * ( LossMatrix[ pair[0] ][j] - LossMatrix[ pair[1] ][j] )
                    if(coef != 0):
                        halfspaceExpr.append( coef * p[j] )
        
        print('halfspaceexpr', sum(halfspaceExpr) )
        if len(halfspaceExpr)>0:
            cs.insert( sum(halfspaceExpr) > 0 )

        polytope = ppl.NNC_Polyhedron(cs)
        if polytope.is_empty() == False:
            P.append(i)

    return P


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
