import numpy as np

## Convex polyhedron manipulation library
import ppl #parma polyhedra library for the cell decomposition
from itertools import islice
import gurobipy as gp
from gurobipy import GRB



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

def get_observer_vector(pair, L,H, S_vectors):

    N, M = H.shape
    Lij = L[pair[0],...] - L[pair[1],...]
    # A = get_alphabet_size(H)
    

    m = gp.Model("mip1")
    m.Params.LogToConsole = 0

    vars = []
    for k in range(N):
        vars.append( [] )
        for a in range( len( set(H[k]) ) ):
            varName =  '{} {}'.format(k,a) 
            vars[k].append( m.addVar( lb = -GRB.INFINITY,  ub = GRB.INFINITY, vtype = GRB.CONTINUOUS,  name = varName) )

    m.update()
    # print('var', vars)

    obj = gp.QuadExpr ()
    for i in range(N):
        for a in range( len( set(H[k]) ) ):
            obj += vars[k][a] * vars[k][a]

    m.setObjective(obj, GRB.MINIMIZE)

    result = 0
    for s,var in zip(S_vectors,vars):
        result += s.dot(var)
    # print('result', result)
    for j in range(M):
        str = 'c {}'.format(j)
        constraintExpr = gp.LinExpr()
        constraintExpr += result[j]
        m.addConstr(constraintExpr == Lij[j], str )

    m.optimize()
    vij =[  np.zeros( len( set(i) ) ) for i in H ] 
    for k in range(N):
        for a in range( len( set(H[k]) ) ):
            # print( 'N', N, 'k',k,'a',a,'var',vars[k][a] )
            vij[k][a] = vars[k][a].X

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
        vars.append( m.addVar(lb = 0 , ub =1 , vtype = GRB.CONTINUOUS, name = varName) )

    m.update()

    # probability simplex constraint:
    simplexExpr = gp.LinExpr()
    for j in range(M):
        simplexExpr += 1.0 * vars[j]
    m.addConstr(simplexExpr == 1.0, "css")

    # the cell decomposition constraint:
    for i3 in range(N):
        if i3 != i1:
            Lij =  LossMatrix[i3] - LossMatrix[i1]
            Lijp = Lij.dot( vars )
            lossConstStr = "cell_constraint {} {}".format(i3, i1) 
            m.addConstr(Lijp >= 0.0, lossConstStr )

    for i3 in range(N):
        if i3 != i2:
            Lij =  LossMatrix[i3] - LossMatrix[i2]
            Lijp = Lij.dot( vars )
            lossConstStr = "cell_constraint {} {}".format(i3, i2) 
            m.addConstr(Lijp >= 0.0, lossConstStr )

    # # two degenerated actions?
    # twoDegenerateExpr = gp.LinExpr()
    # for j in range(M):
    #     twoDegenerateExpr += ( LossMatrix[i2][j]-LossMatrix[i1][j] ) * vars[j]
    # m.addConstr(twoDegenerateExpr == 0.0, "cdeg");

    # for i3 in range(N):
    #     if ( (i3 == i1) or (i2 == i1) ) :
    #         pass 
    #     else:
    #         lossExpr  = gp.LinExpr()
    #         for j in range(M):
    #             lossExpr += ( LossMatrix[i3][j] - LossMatrix[i1][j] ) * vars[j]
    #         lossConstStr = "c{}".format(i3)
    #         m.addConstr(lossExpr >= 0.0, lossConstStr)

    for element in halfspace:
        pair, sign = element[0], element[1]
        if sign != 0:
            Lij = LossMatrix[ pair[0] ] - LossMatrix[ pair[1] ]
            sign_Lij_p = sign * Lij.dot( vars )
            # print( 'halfspace constraint', sign_Lij_p )
            halfspaceConstStr = "halfspace_constraint_{}_{}".format(pair[0],pair[1])
            m.addConstr(sign_Lij_p >= 0.000000000000001,  halfspaceConstStr ) # Gurobi does not support strict inequality constraints therefore constraint >= \epsilon

    # for element in halfspace:
    #     pair, sign = element[0], element[1]
    #     if sign == 0:
    #         pass
    #     else:
    #         halfspaceExpr = gp.LinExpr()
    #         for j in range(M):
    #             halfspaceExpr += ( sign * (LossMatrix[ pair[0] ][j] - LossMatrix[  pair[1]][j] ) ) * vars[j]
        
    #         halfspaceConstStr = "ch_{}_{}".format(pair[0],pair[1])
    #         m.addConstr(halfspaceExpr >= 0.0000000000001,  halfspaceConstStr )

    try: 
        m.optimize()
    except :
        feasible = False

    if m.getAttr( GRB.Attr.Status )  in ( GRB.INF_OR_UNBD , GRB.INFEASIBLE , GRB.UNBOUNDED ):
        feasible = False

    return feasible

def getNeighbors(LossMatrix, halfspace):
    N_t = []
    N, M = LossMatrix.shape
    for i in range(N):
        for j in range(N): 
            if isNeighbor(LossMatrix, i, j, halfspace):
                N_t.append( [i,j] )
    return N_t




def getParetoOptimalActions(LossMatrix, halfspace):

    N, M = LossMatrix.shape
    P = []

    for i in range(N):

        m = gp.Model("mip1")
        m.Params.LogToConsole = 0

        vars = []
        for j in range(M):
            varName =  'p_{}'.format(j) 
            vars.append( m.addVar(lb = 0 , ub =1 , vtype = GRB.CONTINUOUS, name = varName) )

        m.update()
        
        # the variable lies in the probability simplex
        simplexExpr = gp.LinExpr()
        for j in range(M):
            simplexExpr += 1.0 * vars[j]
        m.addConstr(simplexExpr == 1.0, "css")

        # the cell decomposition constraint:
        for i2 in range(N):
            if i2 != i:
                Lij =  LossMatrix[i2] - LossMatrix[i]
                Lijp = Lij.dot( vars )
                lossConstStr = "cell_constraint".format(i2) 
                m.addConstr(Lijp >= 0.0, lossConstStr )

        for element in halfspace:
            pair, sign = element[0], element[1]
            if sign != 0:
                Lij = LossMatrix[ pair[0] ] - LossMatrix[ pair[1] ]
                sign_Lij_p = sign * Lij.dot( vars )
                print( 'halfspace constraint', sign_Lij_p )
                halfspaceConstStr = "halfspace_constraint_{}_{}".format(pair[0],pair[1])
                m.addConstr(sign_Lij_p >= 0.000000000000001,  halfspaceConstStr )  # Gurobi does not support strict inequality constraints therefore constraint >= \epsilon
        try:

            m.optimize()
    
        except gp.GurobiError as e :
            print( 'Error ' )
        except AttributeError as e :
            print ( ' Encountered an attribute error  ')

        if m.getAttr( GRB.Attr.Status )  in ( GRB.INF_OR_UNBD , GRB.INFEASIBLE , GRB.UNBOUNDED ):
            print( 'status action {} is {}'.format(i, m.getAttr( GRB.Attr.Status )))
        else:
            P.append(i)

    return P