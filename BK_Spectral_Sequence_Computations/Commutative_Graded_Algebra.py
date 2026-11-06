import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sympy.polys.matrices import DomainMatrix
from sympy.polys.domains import ZZ
from Labeled_Matrix import LabeledMatrix
from operator import itemgetter
import time
from SSets import SSet, genus_g_surface
from Cohomology_Configuration_Spaces import Cohomology

'''
Goal: compute the E_2 page of the Bousfield-Kan spectral sequence.
For a finite simplicial set X, the simplicial vector space in degree p is
H^*(Conf(X_p,R^D);Q), with differential sum_i (-1)^i d_i.
The target dimension D is ODD and >= 3; generators w_uv have degree D-1.

Initialization:
    -an SSet X, including simplices through the desired horizontal bound
    -an odd dimension D >= 3
    -a vertical bound k: compute q(D-1) for 0<=q<=k
    -direct_normalization: generate normalized basis monomials directly
     (False uses the original full-basis-then-delete construction)

Basis monomials are tuples ((u1,v1),...,(uq,vq)) with ui<vi, sorted pairs
and distinct second endpoints; () denotes the unit in degree zero.
Each d_i returns a list [coefficient, monomial] after applying the face map,
Arnold-Cohen reduction, and projection to the normalized quotient.
'''

class Complex:

    def __init__(self, X, D, k, direct_normalization=True):
        if not isinstance(X, SSet):
            raise TypeError("X must be an SSet.")
        if not isinstance(D,int) or isinstance(D,bool) or D < 3 or D % 2 == 0:
            raise ValueError("D must be odd and >= 3.")
        if not isinstance(k,int) or k < 0:
            raise ValueError("k must be a nonnegative integer.")
        self.sset = X
        self.D = D
        self.k = k
        self._sparse_cache = {}
        self._rank_cache = {}

        if direct_normalization:
            # The same Arnold-Cohen basis as in Cohomology, but enumerate only
            # its nondegenerate monomials. Degenerate partial monomials are
            # still explored: extending them can produce nondegenerate ones.
            self.complex = [Cohomology(X_p,0) for X_p in X.simplices]
            for p, H in enumerate(self.complex):
                H.k = k
                H.basis = [[()] if p == 0 else []]
                for q in range(1,k+1):
                    H.basis.append(self.normalized_basis(p,q,H))
        else:
            self.complex = [Cohomology(X_p,k) for X_p in X.simplices]
            self.remove_all_degenerates()

        self._basis_sets = [[set(basis) for basis in H.basis] for H in self.complex]

    #A monomial is degenerate exactly when all its labels share an index of s_i.
    def is_degenerate_monomial(self, monomial, p=None):
        if not monomial: return p is not None and p > 0  # the unit is degenerate only for p>0
        common = self.sset.degeneracy_indices(monomial[0][0])
        for pair in monomial:
            for label in pair:
                common.intersection_update(self.sset.degeneracy_indices(label))
                if not common: return False
        return bool(common)

    #Construct a normalized Arnold-Cohen basis without first storing the full one.
    def normalized_basis(self, p, q, cohomology=None):
        if q == 0: return [()] if p == 0 else []
        d = len(self.sset.non_degenerate_simplices)-1
        if p > 2*q*d: return []  # each of 2q labels omits at most d degeneracies
        H = cohomology if cohomology is not None else self.complex[p]
        if q >= len(H.S): return []
        pairs = [monomial[0] for monomial in H.get_pairs()]
        masks = {label: sum(1 << i for i in self.sset.degeneracy_indices(label)) for label in H.S}
        pair_masks = [masks[u] & masks[v] for u,v in pairs]
        all_indices = (1 << p)-1

        #A suffix common to every remaining pair cannot be eliminated by
        #appending further pairs; this allows pruning the backtracking tree.
        suffix_and = [all_indices]*(len(pairs)+1)
        for j in range(len(pairs)-1,-1,-1):
            suffix_and[j] = suffix_and[j+1] & pair_masks[j]

        basis = []
        def extend(start, monomial, targets, common):
            remaining = q-len(monomial)
            if remaining == 0:
                if not common: basis.append(monomial)
                return
            if len(pairs)-start < remaining: return
            if common & suffix_and[start]: return
            # A single pair can eliminate at most 2d degeneracy indices.
            if common.bit_count() > 2*d*remaining: return
            for j in range(start,len(pairs)):
                u,v = pairs[j]
                if v in targets: continue
                extend(j+1, monomial+(pairs[j],), targets | {v}, common & pair_masks[j])

        extend(0,(),set(),all_indices)
        return basis

    #Normalization of a previously constructed full basis (kept for comparison).
    def remove_all_degenerates(self):
        for p in range(self.sset.max_dim+1):
            self.complex[p].basis[0] = [()] if p == 0 else []
            for q in range(1,self.k+1):
                self.complex[p].basis[q] = self.remove_degenerates(self.complex,p,q)

    def remove_degenerates(self, copy, p, q):
        #No substring checks: e.g. 11_ is not the same degeneracy as 1_.
        return [chain for chain in copy[p].basis[q] if not self.is_degenerate_monomial(chain)]

    #Sparse matrix for the normalized differential, using the SAME face maps
    #and Arnold reductions as matrix_d. Use this for large computations.
    def matrix_d_sparse(self, p, q):
        if not 0 <= p <= self.sset.max_dim or not 0 <= q <= self.k:
            raise ValueError("p or q outside the computed range.")
        key = (p,q)
        if key in self._sparse_cache: return self._sparse_cache[key]
        col_labels = self.complex[p].basis[q]
        row_labels = [] if p == 0 else self.complex[p-1].basis[q]
        entries = {}
        if row_labels and col_labels:
            row_index = {label: j for j,label in enumerate(row_labels)}
            for col, simplex in enumerate(col_labels):
                for i in range(p+1):
                    for coefficient, monomial in self.d_i(i,simplex):
                        index = (row_index[monomial],col)
                        entries[index] = entries.get(index,0)+(-1)**i*coefficient
        entries = {key:value for key,value in entries.items() if value}
        if entries:
            rows,cols = zip(*entries)
            matrix = coo_matrix((list(entries.values()),(rows,cols)),
                                shape=(len(row_labels),len(col_labels)),dtype=np.int64).tocsr()
        else:
            matrix = csr_matrix((len(row_labels),len(col_labels)),dtype=np.int64)
        self._sparse_cache[key] = matrix
        return matrix

    #Labeled dense version, useful for inspecting individual matrix entries.
    #Do not request this for very large degrees: it materializes a dense array.
    def matrix_d(self, p, q):
        if not 0 <= p <= self.sset.max_dim or not 0 <= q <= self.k:
            raise ValueError("p or q outside the computed range.")
        col_labels = self.complex[p].basis[q]
        row_labels = [] if p == 0 else self.complex[p-1].basis[q]
        return LabeledMatrix(self.matrix_d_sparse(p,q).toarray(),row_labels,col_labels)

    #Matrix of the i-th face followed by projection to normalized chains.
    #Unlike the alternating sum, an individual face need not descend to the quotient.
    def matrix_d_i(self, p, q, i):
        if not 1 <= p <= self.sset.max_dim or not 0 <= i <= p or not 0 <= q <= self.k:
            raise ValueError("Invalid p, q or face index i.")
        row_labels = self.complex[p-1].basis[q]
        col_labels = self.complex[p].basis[q]
        matrix = LabeledMatrix(np.zeros((len(row_labels),len(col_labels)),dtype=np.int64),
                               row_labels,col_labels)
        for col, simplex in enumerate(col_labels):
            if not row_labels: break
            for coefficient, monomial in self.d_i(i,simplex):
                matrix.data[matrix.row_index[monomial],col] += coefficient
        return matrix

    def d_i(self, i, simplex):
        """Face applied to a monomial, Arnold reduction, then normalization."""
        if not simplex:
            raise ValueError("The unit is handled separately in the normalized complex.")
        p = self.sset.dim_simplex(simplex[0][0])
        if not 1 <= p <= self.sset.max_dim or not 0 <= i <= p:
            raise ValueError("Invalid face index or simplicial degree.")
        q = len(simplex)
        d_i_simplex = []
        sgn = 1
        for pair in simplex:
            d_i_u = self.sset.face(i,pair[0])
            d_i_v = self.sset.face(i,pair[1])
            if d_i_v == d_i_u: return [] #w_(uu)=0
            if d_i_u > d_i_v:
                sgn = -sgn  #w_uv=-w_vu, since D is odd
                d_i_u,d_i_v = d_i_v,d_i_u
            d_i_simplex.append((d_i_u,d_i_v))
        chains = [[sgn,tuple(sorted(d_i_simplex))]]
        if len(set(chains[0][1])) != q: return [] #w_(uv)^2=0

        chain_to_split, index_0, index_1 = Complex.some_is_splittable(chains)
        while chain_to_split:
            chains.remove(chain_to_split)
            chains += Complex.split(chain_to_split,index_0,index_1)
            chain_to_split,index_0,index_1 = Complex.some_is_splittable(chains)

        #Collect equal terms: different applications of the Arnold relation
        #may contribute to the same basis monomial, possibly with cancellation.
        coefficients = {}
        for coefficient, monomial in chains:
            if monomial in self._basis_sets[p-1][q]:
                coefficients[monomial] = coefficients.get(monomial,0)+coefficient
            elif not self.is_degenerate_monomial(monomial):
                raise RuntimeError(f"Unrecognized nondegenerate monomial in d_{i}: {monomial}")
        return [[value,monomial] for monomial,value in sorted(coefficients.items()) if value]

    #An array of chains [coefficient, monomial]
    @staticmethod
    def some_is_splittable(chains):
        for chain in chains:
            index_0,index_1 = Complex.is_splittable(chain)
            if index_1 is not False: return chain,index_0,index_1
        return False,False,False

    #Find two factors w_uw and w_vw, u<v<w, to apply the Arnold relation.
    @staticmethod
    def is_splittable(chain):
        k = len(chain[1])
        for index_0 in range(k):
            for index_1 in range(index_0+1,k):
                if chain[1][index_0][1] == chain[1][index_1][1]: return index_0,index_1
        return False,False

    @staticmethod
    def split(chain, index_0, index_1):
        if chain[1][index_0][1] != chain[1][index_1][1]:
            raise ValueError("The two Arnold generators must have a common second endpoint.")
        u,v,w = chain[1][index_0][0], chain[1][index_1][0], chain[1][index_0][1]
        if not u < v < w:
            raise ValueError("Arnold reduction requires u<v<w.")
        #w_uw w_vw = - w_uv w_uw + w_uv w_vw
        remaining = [pair for j,pair in enumerate(chain[1]) if j not in (index_0,index_1)]
        split_0 = tuple(sorted(remaining+[(u,v),(u,w)]))
        split_1 = tuple(sorted(remaining+[(u,v),(v,w)]))
        rtn = []
        if len(set(split_0)) == len(split_0): rtn.append([-chain[0],split_0])
        if len(set(split_1)) == len(split_1): rtn.append([chain[0],split_1])
        return rtn

    def differential_rank(self, p, q, exact=True):
        key = (p,q,exact)
        if key not in self._rank_cache:
            M = self.matrix_d_sparse(p,q)
            if min(M.shape) == 0:
                rank = 0
            elif not exact:
                #Numerical rank is faster for small matrices but may be wrong over Q.
                rank = int(np.linalg.matrix_rank(M.toarray().astype(float)))
            else:
                #Pass the sparse integer entries directly to SymPy, avoiding
                #a dense array (which may not fit in memory at all).
                entries = {}
                coo = M.tocoo()
                for i,j,value in zip(coo.row,coo.col,coo.data):
                    entries.setdefault(int(i),{})[int(j)] = ZZ(int(value))
                rank = int(DomainMatrix(entries,M.shape,ZZ).rank())
            self._rank_cache[key] = rank
        return self._rank_cache[key]

    def e_2_p_q(self, p, q, exact=True):
        if not 0 <= p <= self.sset.max_dim or not 0 <= q <= self.k:
            raise ValueError("p or q outside the computed range.")
        if q == 0: return int(p == 0)
        d = len(self.sset.non_degenerate_simplices)-1
        bound = 2*q*d
        if p > bound: return 0
        if p == self.sset.max_dim and p < bound:
            raise ValueError("E_2 at this top degree requires X_(p+1); increase max_dim.")
        rank_p = self.differential_rank(p,q,exact=exact)
        rank_next = 0 if p >= bound else self.differential_rank(p+1,q,exact=exact)
        betti = len(self.complex[p].basis[q])-rank_p-rank_next
        if betti < 0:
            raise ArithmeticError(f"Negative Betti number at (p,q)=({p},{q}); check differential ranks.")
        return betti

    #Check the main invariant: d_(p-1) d_p = 0, in every requested vertical degree.
    def test_d_squared_zero(self, q=None, sparse=False):
        degrees = range(self.k+1) if q is None else [q]
        for degree in degrees:
            for p in range(2,self.sset.max_dim+1):
                if sparse:
                    product = self.matrix_d_sparse(p-1,degree) @ self.matrix_d_sparse(p,degree)
                    product.eliminate_zeros()
                    if product.nnz:
                        raise AssertionError(f"d^2 != 0 at p={p}, q={degree}.")
                else:
                    A = self.matrix_d(p-1,degree)
                    B = self.matrix_d(p,degree)
                    #Check column by column: the differential is very sparse,
                    #so forming the full matrix product is wasteful.
                    for j in range(B.data.shape[1]):
                        support = np.flatnonzero(B.data[:,j])
                        if support.size and np.any(A.data[:,support] @ B.data[support,j]):
                            raise AssertionError(f"d^2 != 0 at p={p}, q={degree}, column={j}.")
        return True

    #Compare fast construction against the original full-basis normalization.
    #Use only on small examples: the full basis can be enormous.
    def test_normalization(self, p, q):
        full = Cohomology(self.sset.simplices[p],q)
        expected = [()] if q == 0 and p == 0 else []
        if q > 0:
            expected = [monomial for monomial in full.basis[q] if not self.is_degenerate_monomial(monomial)]
        if set(expected) != self._basis_sets[p][q]:
            raise AssertionError((p,q))
        return True

    #Compute and display the E_2 page. Entries not determined by the given
    #horizontal truncation are NaN; known vanishing entries are shown as 0.
    def display_e_2_page(self, exact=True):
        max_p = self.sset.max_dim
        e_2_page = np.full((self.k*(self.D-1)+1,max_p+1),np.nan)
        e_2_page[0,:] = 0
        e_2_page[0,0] = 1
        d = len(self.sset.non_degenerate_simplices)-1
        for q in range(1,self.k+1):
            for p in range(max_p+1):
                if p > 2*q*d:
                    e_2_page[q*(self.D-1),p] = 0
                elif p < max_p or p >= 2*q*d:
                    e_2_page[q*(self.D-1),p] = self.e_2_p_q(p,q,exact=exact)
        result = pd.DataFrame(e_2_page)
        print(str(result))
        return result

#Alternative construction of the double complex
class Free_CGA:

    def __init__(self, generators_with_dimensions, max_word_length) -> None:
        self.generators = list(generators_with_dimensions.keys())
        self.generators.sort()
        self.dimensions = generators_with_dimensions
        self.word_length_filtration = self.fill_word_length_filtration(self.generators,max_word_length)
        self.generators.append('1')
        self.dimensions['1'] = 0
        highest_dim = max(self.dimensions[key] for key in self.dimensions)
        self.grading = [[] for deg in range(highest_dim+1)]
        for word in self.dimensions: self.grading[self.dimensions[word]].append(word)
        
    def tensor(self):
        self.basis = {}
        n = len(self.generators)
        for i in range(n):
            for j in range(i,n):
                a,b = self.generators[i], self.generators[j]
                if i != j or self.dimensions[a] % 2 == 0:
                    tens = '*'.join([a,b])
                    self.basis[tens] = self.dimensions[a] + self.dimensions[b]
        return tens
    
    def fill_word_length_filtration(self,generators,max_word_length):
        filtration = [generators]
        while len(filtration) < max_word_length:
            longest_words = filtration[-1]
            new_words = []
            for letter in generators:
                for word in longest_words:
                    letters = word.split('*')
                    if letter not in letters or self.dimensions[letter] % 2 == 0:
                        letters.append(letter)
                        letters.sort()
                        new_word = '*'.join(letters)
                        if new_word not in new_words: new_words.append(new_word)
                        self.dimensions[new_word] = self.dimensions[letter] + self.dimensions[word]
            filtration.append(new_words)
        return filtration
    
class CGA(Free_CGA):

    def __init__(self, generators_with_dimensions, ideal_generators, max_word_length):
        super().__init__(generators_with_dimensions, max_word_length)
        self.ideal_generators = ideal_generators

        
    

if __name__ == "__main__":

    non_degenerate_simplices = [['v'],['a']]   
    faces = {'a':['v','v']}

    S1 = SSet(non_degenerate_simplices,faces,7)

    start = time.time()

    C = Complex(X=S1,D=5,k=4)

    intermediate_duration = time.time() - start 

    print("Time to prepare the complex is: ", intermediate_duration)

    #for i in range(4): print('matrix_d_{}:'.format(i), C.matrix_d_i(3,1,i))

    #print(C.matrix_d(3,1))

    C.display_e_2_page()

    end_duration = time.time()-intermediate_duration-start

    print("Time to compute the E_2 page: ", end_duration)

    non_degenerate_simplices = [['v'],[],['a']]  
    faces = {'a':['0_v','0_v','0_v']}

    S2 = SSet(non_degenerate_simplices,faces,6)

    #The 2-sphere; note that theoretically speaking, computations do not make sense (we need a parallelizable manifold)
    C2 = Complex(S2,7,1)

    #print(C2.matrix_d(4,1))

    C2.display_e_2_page()

    '''
    non_degenerate_simplices = [['u','v'],['a','b','e0','e1','e2','e3'],['F0','F1','F2','F3']]  
    faces = {'a':['u','u'], 'b':['u','u'], 'e0':['u','v'], 'e1':['u','v'], 'e2':['u','v'], 'e3':['u','v'], 'F0':['b','e0','e1'], 'F1':['a','e1','e2'], 'F2':['b','e3','e2'], 'F3':['a','e0','e3']}

    T = SSet(non_degenerate_simplices,faces,6)

    C3 = Complex(T,5,1)

    C3.display_e_2_page()

    non_degenerate_simplices,faces = genus_g_surface(4)
    T = SSet(non_degenerate_simplices,faces,6)

    C4 = Complex(T,5,1)

    C4.display_e_2_page()
    '''

    #genus g surface; note that theoretically speaking, for g>1, computations do not make sense (we need a parallelizable manifold)
    for g in range(1,3):
        non_degenerate_simplices,faces = genus_g_surface(g)
        T = SSet(non_degenerate_simplices,faces,5)
        C = Complex(T,5,1)
        print('E2 for genus {} surface'.format(g))
        C.display_e_2_page()

    non_degenerate_simplices,faces = genus_g_surface(1)
    #T = SSet(non_degenerate_simplices,faces,8)
    #C = Complex(T,5,2)
    #C.display_e_2_page()
    non_degenerate_simplices = [['v'],['a','b','c'],['D','U']]  
    faces = {'a':['v','v'],'b':['v','v'],'c':['v','v'],'D':['b','c','a'],'U':['a','c','b']}
    T = SSet(non_degenerate_simplices,faces,5)
    C = Complex(T,5,1)
    C.display_e_2_page()

    print("Start the beast")

    #Projective plane; note that theoretically speaking, computations do not make sense (we need a parallelizable manifold)
    non_degenerate_simplices = [['v'],['a'],['F']]
    faces = {'a':['v','v'],'F':['a','0_v','a']}

    P = SSet(non_degenerate_simplices,faces,9)
    C = Complex(P,9,3)

    C.display_e_2_page()