
# import numpy as np
# import ppl

# # return Cell polytope for action i
# def Cell(i,LossMatrix):
#     N, M = LossMatrix.shape

#     # declare M ppl Variables
#     p = [ppl.Variable(j) for j in range(M)]
    
#     # declare polytope constraints
#     cs = ppl.Constraint_System()
    
#     # probabilies constraints on p
#     cs.insert( sum( p[j] for j in range(M)) == 1 )
#     for j in range(M):
#         cs.insert(p[j] >= 0)

#     for j in range(N):
#         Lij = LossMatrix[i,...].T - LossMatrix[j, ...]
#         cs.insert( Lij.T @ p <= 0 )
  
#     return ppl.C_Polyhedron(cs)

# def isDegenerated(i,LossMatrix):
#     N,M = LossMatrix.shape
#     polytope_i = Cell(i,LossMatrix)
#     if polytope_i.is_empty():
#         return False
#     isDegen = False
#     j=0
#     while(not isDegen and j<N):
#         if j!=i:
#             # strict inclusion test
#             if polytope_i < Cell(i,LossMatrix):
#                 #print "Cell(",i,") is strictly inside Cell(", j, ")"
#                 isDegen = True
#         j += 1
#     return isDegen     

# def isNonDominated(i, LossMatrix):
#     return not ( Cell(i,LossMatrix).is_empty())

# def isParetoOptimal(i, LossMatrix):
#     return isNonDominated(i, LossMatrix) and not isDegenerated(i,LossMatrix)


# # Return the polytope where both a and b are winning actions
# def cell_intersection(a, b, LossMatrix):
#     N, M = LossMatrix.shape

#     # declare M ppl Variables
#     p = [ppl.Variable(j) for j in range(M)]
    
#     # declare polytope constraints
#     cs = ppl.Constraint_System()
    
#     # probabilies constraints on p
#     cs.insert( sum( p[j] for j in range(M)) == 1 )
#     for j in range(M):
#         cs.insert(p[j] >= 0)
    
#     for j in range(N):
#         Lij = LossMatrix[a,...].T - LossMatrix[j, ...]
#         cs.insert( Lij.T @ p <= 0 )

#     for j in range(N):
#         Lij = LossMatrix[b,...].T - LossMatrix[j, ...]
#         cs.insert( Lij.T @ p <= 0 )
            
#     return ppl.C_Polyhedron(cs)

# # Check if two actions are neighbours
# def areNeighbours(a, b, LossMatrix):
#     M = len(LossMatrix[0])
#     return cell_intersection(a, b, LossMatrix).affine_dimension() >= M - 2

# def neighborhood(a,b, LossMatrix):
#     N, M = LossMatrix.shape
#     result = []
#     intersection = cell_intersection(a,b,LossMatrix)
#     for k in range(N):
#         cell_k = Cell(k,LossMatrix)
#         if intersection <= cell_k:
#             result.append(k)
#     return result
