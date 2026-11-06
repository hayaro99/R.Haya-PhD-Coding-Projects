#Homemade implementation of simplicial set
from Labeled_Matrix import LabeledMatrix
import numpy as np
import time

'''
Implementation of finite simplicial sets:
-Initialization: 
    -specify the non degenerate simplices in a double array non_degenerate_simplices of length d+1 where d=dimension
    non_degenerate_simplices[p]=array of non degenerate p-simplices
    format of a non degenerate simplex: all characters except _
    Example: non_degenerate_simplices = [['0','1','2'],['01','02','12'],['012']] (standard 2-simplex)
    -specify the face relations in a dictionary whose keys are non degenerate simplices and values are arrays of faces
    Example: faces = {'01':['1','0'],'02':['2','0'],'12':['2','1'],'012':['12','02','01']} (NB: don't specify for 0-simplices)
    -max_dim parameter: truncate the simplicial set (max_dim-skeleton)

-Degeneracies:
    -format: for a nondegenerate simplex x, the degeneracy s_ik(...(s_i0(x))) is written ik_..._i0_x with ik>...>i0

'''

class SSet():

    def __init__(self, non_degenerate_simplices, faces, max_dim=-1) -> None:
        if not non_degenerate_simplices:
            raise ValueError("Specify at least X_0 (possibly empty).")
        if max_dim < -1:
            raise ValueError("max_dim must be >= -1.")
        self.non_degenerate_simplices = [list(level) for level in non_degenerate_simplices]
        self.faces = dict(faces)
        self.max_dim = max(max_dim,len(non_degenerate_simplices)-1)
        self.simplices = self.get_all_simplices_up_to_dim_n(self.max_dim)
        self.validate_faces()

    #Checks whether or not a simplex is degenerate
    def is_degenerate(self,simplex):
        return '_' in simplex 
    
    def degeneracy(self,i,simplex):
        if not self.is_degenerate(simplex): return str(i)+'_'+simplex
        s_simplex = str(i)+'_'+simplex 
        degeneracies_indices, non_degen_simplex = s_simplex.rsplit('_',1)
        degeneracies_indices = self.order_list(self.chain_to_list(degeneracies_indices))
        return self.list_to_chain(degeneracies_indices)+'_'+non_degen_simplex

    #A string ik_..._i0 into a list [ik,...,i0]
    def chain_to_list(self,chain):
        try:
            return list(map(int,chain.split('_')))
        except ValueError as exc:
            raise ValueError('Invalid degeneracy indices: '+chain) from exc

    #A list [ik,...,i0] into a string ik_..._i0
    def list_to_chain(self,arr):
        return '_'.join(map(str,arr))

    #Ensures the order ik>...>i0
    def order_list(self,arr):
        n = len(arr)
        to_process = True
        while to_process:
            to_process = False
            for index in range(n-1):
                if arr[index] <= arr[index+1]: 
                    to_process = True
                    arr[index], arr[index+1] = arr[index+1]+1, arr[index]
        return arr

    #Returns the i-th face of simplex (beware that a nondegenerate simplex can have degenerate faces)
    def face(self,i,simplex):
        if not self.is_degenerate(simplex): return self.faces[simplex][i]
        top_degen_index, less_degen_simplex = simplex.split('_',1)
        top_degen_index = int(top_degen_index)
        if i < top_degen_index: return self.degeneracy(top_degen_index-1,self.face(i,less_degen_simplex))
        if i > top_degen_index+1: return self.degeneracy(top_degen_index,self.face(i-1,less_degen_simplex))
        return less_degen_simplex

    #Returns an array of sets of all simplices until degree n
    def get_all_simplices_up_to_dim_n(self,n):
        all_simplices = []
        X_0 = set(self.non_degenerate_simplices[0])
        all_simplices.append(X_0)
        for p in range(1,n+1):
            X_p = set() if p >= len(self.non_degenerate_simplices) else set(self.non_degenerate_simplices[p])
            for simplex in all_simplices[-1]:
                for i in range(p):
                    X_p.add(self.degeneracy(i,simplex))
            all_simplices.append(X_p)
        return all_simplices

    def degeneracy_indices(self, simplex):
        """Indices in the canonical Eilenberg-Zilber degeneracy expression."""
        if not self.is_degenerate(simplex): return set()
        return set(self.chain_to_list(simplex.rsplit('_',1)[0]))

    def validate_faces(self):
        """Check the input nondegenerate face data and their simplicial identities."""
        names = [x for level in self.non_degenerate_simplices for x in level]
        if any(not isinstance(x,str) or '_' in x for x in names):
            raise ValueError("Nondegenerate labels must be strings without '_'.")
        if len(set(names)) != len(names):
            raise ValueError("Nondegenerate simplex labels must be unique across dimensions.")
        required = set(names) - set(self.non_degenerate_simplices[0])
        if set(self.faces) != required:
            raise ValueError("Supply face relations for exactly the positive-dimensional nondegenerate simplices.")
        for p in range(1,len(self.non_degenerate_simplices)):
            for x in self.non_degenerate_simplices[p]:
                if len(self.faces[x]) != p+1 or any(y not in self.simplices[p-1] for y in self.faces[x]):
                    raise ValueError(f"Invalid list of faces for {x!r} in dimension {p}.")
                if p >= 2:
                    for i in range(p+1):
                        for j in range(i+1,p+1):
                            if self.face(i,self.face(j,x)) != self.face(j-1,self.face(i,x)):
                                raise ValueError(f"Simplicial face identity fails on {x!r} for i={i}, j={j}.")
        return True

    def test_simplicial_identities(self):
        """Check all face, degeneracy and mixed identities through max_dim."""
        for p, level in enumerate(self.simplices):
            for x in level:
                if p >= 2:
                    for i in range(p+1):
                        for j in range(i+1,p+1):
                            if self.face(i,self.face(j,x)) != self.face(j-1,self.face(i,x)):
                                raise AssertionError((x,i,j))
                if p+2 <= self.max_dim:
                    for i in range(p+1):
                        for j in range(i,p+1):
                            if self.degeneracy(i,self.degeneracy(j,x)) != self.degeneracy(j+1,self.degeneracy(i,x)):
                                raise AssertionError((x,i,j))
                if p+1 <= self.max_dim:
                    for j in range(p+1):
                        y = self.degeneracy(j,x)
                        for i in range(p+2):
                            if i < j:
                                expected = self.degeneracy(j-1,self.face(i,x))
                            elif i in (j,j+1):
                                expected = x
                            else:
                                expected = self.degeneracy(j,self.face(i-1,x))
                            if self.face(i,y) != expected:
                                raise AssertionError((x,i,j))
        return True

    #Dold-Kan correspondance matrix: computes the differential d:X_p->X_{p-1}, sum_{i=0}^p (-1)^i d_i
    def matrix_d(self, p):
        if not 0 <= p < len(self.non_degenerate_simplices):
            raise ValueError("p is outside the nondegenerate simplicial range.")
        row_labels = [] if p == 0 else self.non_degenerate_simplices[p-1]
        col_labels = self.non_degenerate_simplices[p]
        if p == 0 or not row_labels: return LabeledMatrix(np.zeros((len(row_labels),len(col_labels)),dtype=int),row_labels,col_labels)
        matrix = LabeledMatrix(np.zeros((len(row_labels), len(col_labels)),dtype=int),
                               row_labels,
                               col_labels)
        sgn = 1
        for i in range(p+1):
            matrix_i = self.matrix_d_i(p,i)
            matrix += matrix_i*sgn
            sgn = -sgn 
        return matrix

    #Matrix for the map d_i: X_p->X_{p-1}
    def matrix_d_i(self, p, i):
        if not 1 <= p < len(self.non_degenerate_simplices) or not 0 <= i <= p:
            raise ValueError("Invalid p or face index i.")
        row_labels = self.non_degenerate_simplices[p-1]
        col_labels = self.non_degenerate_simplices[p]
        if not row_labels: return LabeledMatrix(np.zeros((0,len(col_labels)),dtype=int),row_labels,col_labels)
        matrix = LabeledMatrix(np.zeros((len(row_labels), len(col_labels)),dtype=int),
                               row_labels,
                               col_labels)
        for simplex in col_labels:
            d_i_simplex = self.face(i,simplex)
            if not self.is_degenerate(d_i_simplex): matrix[d_i_simplex,simplex] = 1
        return matrix

    #p-th Betti number of a simplicial abelian group
    def betti(self,p):
        differential_p = self.matrix_d(p)
        if p == len(self.non_degenerate_simplices)-1: return len(differential_p.col_labels)-differential_p.rank()
        differential_p_1 = self.matrix_d(p+1)
        return len(differential_p.col_labels)-differential_p.rank()-differential_p_1.rank()

    #Dimension of a (possibly degenerate) simplex
    def dim_simplex(self,simplex):
        parts = simplex.split('_')
        non_degen_dimension = 0 if parts[-1] in self.non_degenerate_simplices[0] else len(self.faces[parts[-1]])-1
        return len(parts) - 1 + non_degen_dimension

#A model of the genus_g_surface
def genus_g_surface(g):
    edges = ['a'+str(i) for i in range(g)]+['b'+str(i) for i in range(g)]+['e'+str(i) for i in range(4*g)]
    faces = ['F'+str(i) for i in range(4*g)]
    non_degenerate_simplices = [['u','v'], edges, faces]
    faces_relations = {}
    for i in range(g): 
        faces_relations['a'+str(i)] = ['v','v']
        faces_relations['b'+str(i)] = ['v','v']
    for i in range(4*g):
        faces_relations['e'+str(i)] = ['v','u']
        if i%4 == 0: faces_relations['F'+str(i)] = ['a'+str(i//4),'e'+str(i+1),'e'+str(i)]
        elif i%4 == 1: faces_relations['F'+str(i)] = ['b'+str(i//4),'e'+str(i+1),'e'+str(i)]
        elif i%4 == 2: faces_relations['F'+str(i)] = ['a'+str(i//4),'e'+str(i),'e'+str(i+1)]
        else: faces_relations['F'+str(i)] = ['b'+str(i//4),'e'+str(i),'e'+str((i+1)%4)]
    return non_degenerate_simplices, faces_relations



if __name__ == "__main__":
    begin = time.time()

    #Standard 2-simplex
    non_degenerate_simplices = [['0','1','2'],['01','02','12'],['012']]   
    faces = {'01':['1','0'],'02':['2','0'],'12':['2','1'],'012':['12','02','01']}

    Delta_2 = SSet(non_degenerate_simplices,faces)

    for p in range(3): print(Delta_2.betti(p))
    print('----------------')

    #The circle
    non_degenerate_simplices = [['v'],['a']]   
    faces = {'a':['v','v']}

    S1 = SSet(non_degenerate_simplices,faces)
    print("test face", S1.face(0,'0_v'))

    for p in range(2): print(S1.betti(p))
    print('----------------')

    #Minimal model of the sphere S^2; note that there is no nondegenerate 1-simplex
    non_degenerate_simplices = [['0'],[],['a']]  
    faces = {'a':['0_0','0_0','0_0']}

    S2 = SSet(non_degenerate_simplices,faces)

    for p in range(3): print(S2.betti(p))
    print('----------------')

    #The torus
    non_degenerate_simplices = [['u','v'],['a','b','e0','e1','e2','e3'],['F0','F1','F2','F3']]  
    faces = {'a':['u','u'], 'b':['u','u'], 'e0':['u','v'], 'e1':['u','v'], 'e2':['u','v'], 'e3':['u','v'], 'F0':['b','e0','e1'], 'F1':['a','e1','e2'], 'F2':['b','e3','e2'], 'F3':['a','e0','e3']}

    T = SSet(non_degenerate_simplices,faces)

    for p in range(3): print(T.betti(p),len(T.simplices[p]))
    print('---------------')

    non_degenerate_simplices, faces = genus_g_surface(5)
    T = SSet(non_degenerate_simplices,faces)
    for p in range(3): print(T.betti(p),len(T.simplices[p]))

    print("Duration: ", time.time()-begin, " seconds")

    #Projective plane
    non_degenerate_simplices = [['0'],['a'],['F']]
    faces = {'a':['0','0'],'F':['a','0_0','a']}
    P = SSet(non_degenerate_simplices,faces,7)
    for p in range(3): print(P.betti(p),len(P.simplices[p]))
