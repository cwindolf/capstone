import numpy as np
from tqdm import trange
from functools import reduce


# *************************************************************************** #
# PGF helper fns
# *************************************************************************** #

def pgf_expectation(g):
    return g.dds()(1)


def pgf_predict_mean_degree(p, r, f, m):
    g_p, g_r = single_pgf(p, 'g_p'), single_pgf(r, 'g_r')
    gzm = [composition_of_pgfs([g_p] + [g_r] * i) for i in range(m)]
    gfm = [bernoulli_pgf(f(i), 'g_f%d' % i) for i in range(m)]
    return sum(gfm[i].dds()(1) * gzm[i].dds()(gfm[i](1))
               for i in range(m))


def generate_pmf(g, m):
    '''
    given a pgf g, return the pmf p=(p_0, ..., p_m)
    '''
    m += 1
    p = np.zeros((m,), dtype=g.dtype)
    p[0] = g(0)
    fact = 1
    for i in range(1, m):
        print(i)
        print(g)
        # print(g)
        fact *= i
        g = g.dds()
        p[i] = g(0) / fact
    return p

def bernoulli_pgf(p, name):
    return single_pgf([1 - p, p], name=name)


# *************************************************************************** #
# PGF classes to handle differentiation.
# *************************************************************************** #

class single_pgf():

    def __init__(self, a, name='g_a'):
        '''
        for a sequence a=(a_0, a_1, ...), return a's generating function.
        '''
        self.a = np.asarray(a)
        self.k = np.arange(self.a.shape[0])
        self.name = name

    def __call__(self, s):
        return np.dot(self.a, np.power(s, self.k))

    def dds(self):
        if self.a.shape[0] == 1:
            return None
        else:
            return single_pgf((self.a * self.k)[1:], name=('%s\'' % self.name))

    @property
    def dtype(self):
        return self.a.dtype

    def __str__(self):
        return self.name
    


class sum_of_pgfs():

    def __init__(self, pgfs):
        self.pgfs = pgfs

    def __call__(self, s):
        return sum(g(s) for g in self.pgfs)

    def dds(self):
        new_pgfs = []
        for g in self.pgfs:
            dds_g = g.dds()
            if dds_g:
                new_pgfs.append(dds_g)
        if new_pgfs:
            return sum_of_pgfs(new_pgfs)
        else:
            return None

    @property
    def dtype(self):
        return self.pgfs[0].dtype

    def __str__(self):
        return ('(' + ' + '.join(map(str, self.pgfs)) + ')')


class product_of_pgfs():

    def __init__(self, pgfs):
        self.pgfs = pgfs

    def __call__(self, s):
        return reduce(np.multiply, (g(s) for g in self.pgfs))

    def dds(self):
        # product rule
        new_products = []
        for i, g in enumerate(self.pgfs):
            dds_g = g.dds()
            if dds_g:
                new_products.append(product_of_pgfs(self.pgfs[:i] + [dds_g] + self.pgfs[i+1:]))
        if new_products:
            return sum_of_pgfs(new_products)
        else:
            return None

    @property
    def dtype(self):
        return self.pgfs[0].dtype

    def __str__(self):
        return ('(' + ' * '.join(map(str, self.pgfs)) + ')')


class composition_of_pgfs():

    def __init__(self, pgfs):
        self.pgfs = pgfs
        self.rpgfs = reversed(pgfs)

    def __call__(self, s):
        return reduce(lambda s, g: g(s), self.rpgfs, s)

    def dds(self):
        # chain rule
        new_compositions = []
        for i, g in enumerate(self.pgfs):
            dds_g = g.dds()
            if dds_g:
                new_compositions.append(composition_of_pgfs([dds_g] + self.pgfs[i+1:]))
        if new_compositions:
            return product_of_pgfs(new_compositions)
        else:
            return None

    @property
    def dtype(self):
        return self.pgfs[0].dtype

    def __str__(self):
        return ('(' + ' ￮ '.join(map(str, self.pgfs)) + ')')
