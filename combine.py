import inspect
from collections import OrderedDict


def get_names(sig):
    names = [
        (name, value)
        for name, value in sig.parameters.items()
        if value.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    return OrderedDict(names)


def filter_kwargs(sig, names, kwargs):
    names_in_kwargs = [(name, value) for name, value in kwargs.items() if name in names]
    return OrderedDict(names_in_kwargs)


def skip_pars(names1, names2, num_skipped):
    skipped_pars1 = list(names1.keys())[:num_skipped]
    skipped_pars2 = list(names2.keys())[:num_skipped]
    if skipped_pars1 == skipped_pars2:
        pars1 = list(names1.values())[num_skipped:]
        pars2 = list(names2.values())[num_skipped:]
    else:
        raise Exception("First {} arguments " "have to be the same".format(num_skipped))
    return pars1, pars2


def combine(f, g, operator, num_skipped=0):
    if not callable(f) or not callable(g):
        raise Exception("One of the functions is not a function")

    sig1 = inspect.signature(f)
    sig2 = inspect.signature(g)

    names1 = get_names(sig1)
    names2 = get_names(sig2)

    pars1, pars2 = skip_pars(names1, names2, num_skipped)
    skipped_pars = list(names1.values())[:num_skipped]

    pars1_names = {p.name for p in pars1}
    pars2 = [p for p in pars2 if p.name not in pars1_names]

    parameters = pars1 + pars2
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    parameters = [p.replace(kind=kind) for p in parameters]
    parameters = skipped_pars + parameters

    def wrapped(*args):
        d = {p.name: arg for arg, p in zip(args, parameters)}
        fval = f(*[d[name] for name in names1.keys()])
        gval = g(*[d[name] for name in names2.keys()])
        return operator(fval, gval)

    wrapped.__signature__ = inspect.Signature(parameters=parameters)
    return wrapped
