'''
Encodes the rational cohomology ring of Conf(S,R^D) for D ODD and >= 3.
Based on the Arnold-Cohen description: generators w_ij in degree D-1,
with w_ij=-w_ji, w_ii=w_ij^2=0 and the three-term relations.
For D odd the generators have even degree, hence commute.

-Initialization:
    -an array S of distinct sortable labels -> fixes an order.
    -an integer k: compute degrees 0,(D-1),...,k(D-1).

-A basis monomial is a tuple ((u1,v1),...,(uq,vq)), sorted by pairs,
 with ui<vi and with pairwise distinct second endpoints vi.
 The empty tuple () represents the unit in H^0.
'''

class Cohomology:

    #Object to compute H^*(Conf(S,R^D)) up to degree k(D-1)
    def __init__(self,S,k) -> None:
        if k < 0:
            raise ValueError("k must be nonnegative.")
        self.S = list(S)
        if len(set(self.S)) != len(self.S):
            raise ValueError("S must contain distinct labels.")
        self.S.sort()
        self.k = k
        self.basis = [[()]]  # H^0, including when S is empty
        if k >= 1:
            self.basis.append(self.get_pairs())
        for do in range(1,k):
            # H^q vanishes for q >= |S|; avoid generating empty bases.
            self.basis.append(self.get_basis() if do+1 < len(self.S) else [])

    #Get all generators w_ij with i<j in S
    def get_pairs(self):
        pairs = []
        n = len(self.S)
        for i in range(n):
            for j in range(i+1,n): pairs.append(((self.S[i],self.S[j]),))
        return pairs

    #Computes a basis of degree (i+1)(D-1) from the degree i(D-1) basis
    def get_basis(self):
        basis_1, basis_i = self.basis[1], self.basis[-1]
        new_basis = []
        for i_pairs in basis_i:
            for one_pair in basis_1:
                if one_pair[0] > i_pairs[-1]:
                    b = one_pair[0][1]
                    if all(b != pair[1] for pair in i_pairs):
                        new_basis.append(i_pairs + one_pair)
        return new_basis


if __name__ == "__main__":
    from SSets import SSet

    non_degenerate_simplices = [['0'],['a']]
    faces = {'a':['0','0']}
    S1 = SSet(non_degenerate_simplices,faces,4)
    for p in range(4):
        print(p, S1.simplices[p])
    H = Cohomology(list(S1.simplices[3]),2)
    print(H.S)
    for i in range(H.k+1): print(H.basis[i])
    print(S1.degeneracy(0,'0_a'))
