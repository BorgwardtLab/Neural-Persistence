import collections
import math

import numpy as np
import aleph as al


class PerLayerCalculation:
    """
    Calculates total persistence for a set of weight matrices. Each
    weight matrix is considered to describe a bipartite graph, i.e.
    an individual layer.
    """

    def __call__(self, weights):
        """
        Given a set of weights (supplied as an ordered dictionary) for
        some layers, performs normalization and transforms each of the
        layers into a bipartite graph.
        """

        # Pre-process weights in order to ensure that they already satisfy
        # the filtration constraints correctly.

        W_min = min( [ np.min(weight) for _, weight in weights.items() ] )
        W_max = max( [ np.max(weight) for _, weight in weights.items() ] )
        W     = max( abs(W_min), abs(W_max) )

        # Will store individual total persistence values (normalized and
        # regular ones) per layer, including a total persistence for the
        # complete network, which is the sum of the layers.
        result                                   = collections.defaultdict(dict)
        accumulated_total_persistence            = 0.0
        accumulated_total_persistence_normalized = 0.0

        for name, M in weights.items():
            D  = al.calculateZeroDimensionalPersistenceDiagramOfMatrix( \
                    np.abs(M) / W,                                      \
                    reverseFiltration = True,                           \
                    vertexWeight      = 1.0,                            \
                    unpairedData      = 0.0)
            tp = al.norms.pNorm(D)
            n  = M.shape[0] + M.shape[1]

            result[name]["total_persistence"]            = tp
            result[name]["total_persistence_normalized"] = tp / math.sqrt(n-1)

            accumulated_total_persistence            += tp
            accumulated_total_persistence_normalized += tp / math.sqrt(n-1)

        result["global"]["accumulated_total_persistence"]            = accumulated_total_persistence
        result["global"]["accumulated_total_persistence_normalized"] = accumulated_total_persistence_normalized / sum( 1 for M in weights.values() )

        return result

    def _to_simplicial_complex(self, M):
        """
        Converts an adjacency matrix to a simplicial complex. Does not
        change the weights and will apply a default filtration.
        """

        n,m = M.shape

        # Vertices
        vertices = []
        for i in range(n+m):
            vertices.append( al.Simplex([i], 1.0) )

        # Edges
        edges = []
        for i in range(n):
            for j in range(m):
                edges.append( al.Simplex([i,j+n], M[i,j]) )

        edges = sorted(edges, key = lambda x: x.data, reverse=True)

        # Simplicial complex
        K = al.SimplicialComplex()

        for vertex in vertices:
            K.append(vertex)

        for edge in edges:
            K.append(edge)

        return K

    def _calculate_persistence_diagram_and_pairing(self, K):
        """
        Calculates a zero-dimensional persistence diagram and its
        corresponding persistence pairing. The pairing is used to
        assign total persistence values per vertex, if desired by
        the client.
        """

        # Notice that the second argument ensures that the diagram will
        # not contain any unpaired points. This is tested afterwards in
        # order to ensure that the conversion worked.
        pd, pp = al.calculateZeroDimensionalPersistenceDiagramAndPairing(K, 0.0)

        assert pd.betti == 0
        return pd, pp

