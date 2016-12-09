# haskell-style lazy list
import itertools as it

def conslist(iterable):
    iterator = iterable.__iter__()
    def next_conslist():
        # ceci n'est pas une recursive function
        return lazycons(iterator.next(), next_conslist)

    try:
        return next_conslist()
    except StopIteration:
        return []

def conslistgen(generator_function):
    def conslist_function(*args, **kwargs):
        return conslist(generator_function(*args, **kwargs))
    return conslist_function

class thunk(object):
    def __init__(self, delayed_value):
        self._delayed_value = delayed_value
        self._value = None

    def value(self):
        if self._value is None:
            self._value = self._delayed_value()
        return self._value

def cons(head, tail):
    return lazycons(head, lambda: tail)
        
class lazycons(object):
    def __init__(self, head, delayed_tail, N=None):
        self._head = head
        self._tail = thunk(delayed_tail).value  # remains unevaluated until called

    def __getitem__(self, i):
        if i == 0:
            return self._head
        elif i == slice(1, None, None):
            return self._tail()
        elif isinstance(i, int):
            if i < 0:
                return list(self)[i]
            else:
                return self[i:][0]
        elif isinstance(i, slice):
            return islice(self, i.start, i.stop, i.step)
        else:
            raise Exception("Can't index conslist with {}".format(i))

    def __iter__(self):
        tail = self
        while True:
            yield tail[0]
            tail = tail[1:]  # may raise StopIteration

def iunzip(L, N=2):
    return [imap(nth(i), L) for i in range(N)]

# These extend the functionality of the itertools version
imap = conslistgen(it.imap)
izip = conslistgen(it.izip)
islice = conslistgen(it.islice)
count = conslistgen(it.count)
repeat = conslistgen(it.repeat)
chain = conslistgen(it.chain)

def head(x): return x[0]
def tail(x): return x[1:]
def nth(i): return lambda x: x[i]

@conslistgen
def scanl(f, accumulator, *lists):
    for items in izip(*lists):
        accumulator = f(accumulator, *items)
        yield accumulator

@conslistgen
def tails(xs):
    while True:
        yield xs
        xs = tail(xs)
@conslistgen
def tails(xs):
    while True:
        yield xs
        xs = tail(xs)

@conslistgen
def inits(xs):
    x_init = []
    yield x_init
    for x in xs:
        x_init = x_init + [x]
        yield x_init
