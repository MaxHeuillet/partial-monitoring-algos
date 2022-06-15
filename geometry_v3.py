import numpy as np
import gurobipy as gp
from gurobipy import GRB
import collections

def alphabet_size(FeedbackMatrix, N,M):
    alphabet = []
    for i in range(N):
        for j in range(M):
            alphabet.append(FeedbackMatrix[i][j])
    return len(set(alphabet)) 

def calculate_signal_matrices(FeedbackMatrix, N,M,A):
    signal_matrices = []
    for i in range(N):
        signalMatrix = np.zeros( (A,M) )
        for j in range(M):
            a = FeedbackMatrix[i][j]
            signalMatrix[a][j] = 1
        signal_matrices.append(signalMatrix)
    return signal_matrices
def getParetoOptimalActions(LossMatrix, N, M, halfspace):
    actions = []
    for z in range(N):
        feasible = True

        # try:
        m = gp.Model( )
        m.Params.LogToConsole = 0

        vars = []
        for i in range(M):
            varName =  'p_{}'.format(i) 
            vars.append( m.addVar(0.00001, 1.0, -1.0, GRB.CONTINUOUS, varName) )
            m.update()
        # print('vars pareto',vars)

        # the variable lies in the probability simplex
        simplexExpr = gp.LinExpr()
        for j in range(M):
            simplexExpr += 1.0 * vars[j]

        m.addConstr(simplexExpr == 1.0, "css")

        for i2 in range(N):
            if(i2 != z):
                lossExpr = gp.LinExpr()
                for j in range(M):
                    lossExpr += ( LossMatrix[i2][j] - LossMatrix[z][j] ) * vars[j]
                    lossConstStr = 'c {}'.format(i2)
                m.addConstr(lossExpr >= 0.0, lossConstStr )

        for element in halfspace:
            pair, sign = element[0], element[1]
            if sign != 0:
                halfspaceExpr = gp.LinExpr()
                for j in range(M):
                    coef = sign * (LossMatrix[ pair[0] ][j]-LossMatrix[  pair[1] ][j] ) 
                    if coef != 0:
                        halfspaceExpr += coef * vars[j]
                halfspaceConstStr = "ch_{}_{}".format( pair[0] ,pair[1] )
                m.addConstr(halfspaceExpr >= 0.001, halfspaceConstStr )
        try:
            m.optimize()
            # print('action',z,'status',m.Status)
            objval = m.objVal
        except:
            # print('pareto declined action {}'.format(i))
            feasible=False

        if feasible:
            actions.append(z)

    return actions

def getNeighborhoodActions(LossMatrix, N, M, halfspace,mathcal_N):
    actions = []
    for pair in mathcal_N:
        i1,i2 = pair
        if isNeighbor(LossMatrix, N, M, i1, i2, halfspace):
            actions.append( [i1,i2] )
    return actions

def isNeighbor(LossMatrix, N, M, i1, i2, halfspace):
    feasible = True


    m = gp.Model( )
    m.Params.LogToConsole = 0
    vars = []
    for j in range(M):
        varName = "p {}".format(j)
        vars.append( m.addVar(0.00001, 1.0, -1.0, GRB.CONTINUOUS, varName ) )
        m.update()

    simplexExpr = gp.LinExpr()
    for j in range(M):
        simplexExpr += 1.0 * vars[j]
    m.addConstr(simplexExpr == 1.0, "css") 

    twoDegenerateExpr = gp.LinExpr()
    for j in range(M):
        twoDegenerateExpr += (LossMatrix[i2][j]-LossMatrix[i1][j]) * vars[j]
    m.addConstr(twoDegenerateExpr == 0.0, "cdeg")

    for i3 in range(N):
        if( (i3 == i1) or (i2 == i1) ):
            pass
        else:
            lossExpr = gp.LinExpr()
            for j in range(M):
                lossExpr += ( LossMatrix[i3][j]-LossMatrix[i1][j] ) * vars[j]
            lossConstStr = "c".format(i3)
            m.addConstr(lossExpr >= 0.0, lossConstStr )

    for element in halfspace:
        pair, sign = element[0], element[1]
        if sign != 0:
            halfspaceExpr = gp.LinExpr()
            for j in range(M):
                coef = sign * (LossMatrix[ pair[0] ][j]-LossMatrix[  pair[1] ][j] ) 
                if coef != 0:
                    halfspaceExpr += coef * vars[j]
            halfspaceConstStr = "ch_{}_{}".format( pair[0] ,pair[1] )
            m.addConstr(halfspaceExpr >= 0.001, halfspaceConstStr )
    try:
        m.optimize()
        objval = m.objVal
    except:
        # print('neighbors rejects pair {}{}'.format(i1,i2) )
        feasible = False

    return feasible


def getV(LossMatrix, N, M, A, SignalMatrices, neighborhood_actions):
    v = collections.defaultdict(dict)
    # v[0][0] = [ np.array([[0]]) ]
    # v[1][1] = [ None, np.array([[0,0]]) ]
    # v[0][1] = [  np.array([[0]]), np.array([[-1, 1]])  ]
    # v[1][0] = [  np.array([[0]]), np.array([[1, -1]])  ]
    v[0][0] = [ np.array([[0]]) ]
    v[1][1] = [ None, np.array([[0,0]]) ]
    v[0][1] = [  np.array([[0.5]]), np.array([[-1.5, 0.5]])  ]
    v[1][0] = [  np.array([[0.5]]), np.array([[0.5, -1.5]])  ]
    
    # for pair in neighborhood_actions:
    #     v[ pair[0] ][ pair[1] ]  = getVij(LossMatrix, N, M, A, SignalMatrices, pair[0], pair[1])
    return v
  
# def getVij(LossMatrix, N, M, A, SignalMatrices, i1, i2):
#     l1 = LossMatrix[i1]
#     l2 = LossMatrix[i2]
#     ldiff = l1 - l2
#     # try:
#     m = gp.Model( )
#     m.Params.LogToConsole = 0

#     vars = []

#     for k in range(N):
#         vars.append([])
#         for a in range(A):
#             varName = "v_{}_{}".format(k,a) 
#             vars[k].append( m.addVar(-GRB.INFINITY, GRB.INFINITY, 0., GRB.CONTINUOUS, varName ) ) 
#             m.update()

#     # print('vars', vars)
#     #m.update()

#     obj = 0
#     for k in range(N):
#         for a in range(A):
#             obj += vars[k][a]**2
#     # print('objective', obj )
#     m.setObjective(obj, GRB.MINIMIZE)

#     for j in range(M):
#         constraintExpr = gp.LinExpr()
#         constraintStr = "c_".format(j)
#         for a in range(A):
#             for k in range(N):
#                 constraintExpr += SignalMatrices[k][a][j] * vars[k][a]
#         m.addConstr( constraintExpr == ldiff[j],  constraintStr)
#     # print('model', m)
#     m.optimize()

#     vij = np.zeros( (N, A) )
#     for k in range(N):
#         for a in range(A):
#             # print( vars[k][a] )
#             vij[k][a] =  vars[k][a].X 

#     return vij

#     # except:
#     #     print('error in vij')


def getConfidenceWidth( neighborhoodActions, N_plus, v,  N):
    W = np.zeros(N)

    for pair in neighborhoodActions:
        for k in N_plus[ pair[0] ][ pair[1] ]:
            vec = v[ pair[0] ][ pair[1] ][k]
            # print('vec', vec, 'norm', np.linalg.norm(vec, np.inf) )
            W[k] = np.max( [ W[k], np.linalg.norm(vec, np.inf) ] )
    return W
  
def f(t, alpha):
    return pow(alpha*t*t*np.log(t), 1/3)


  
        
        

