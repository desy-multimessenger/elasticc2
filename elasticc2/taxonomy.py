"""
Implementation of the elasticc2 taxonomy tree:

Alert
├── 0 Meta
│   ├── 100 Meta/Other
│   ├── 200 Residual
│   └── 300 NotClassified
├── 1000 Static
│   └── 1100 Static/Other
└── 2000 Variable
    ├── 2100 Variable/Other
    ├── 2200 Non-Recurring
    │   ├── 2210 Non-Recurring/Other
    │   ├── 2220 SN-like
    │   │   ├── 2221 SN-like/Other
    │   │   ├── 2222 Ia
    │   │   ├── 2223 Ib/c
    │   │   ├── 2224 II
    │   │   ├── 2225 Iax
    │   │   └── 2226 91bg
    │   ├── 2230 Fast
    │   │   ├── 2231 Fast/Other
    │   │   ├── 2232 KN
    │   │   ├── 2233 M-dwarf Flare
    │   │   ├── 2234 Dwarf Novae
    │   │   └── 2235 uLens
    │   └── 2240 Long
    │       ├── 2241 Long/Other
    │       ├── 2242 SLSN
    │       ├── 2243 TDE
    │       ├── 2244 ILOT
    │       ├── 2245 CART
    │       └── 2246 PISN
    └── 2300 Recurring
        ├── 2310 Recurring/Other
        ├── 2320 Periodic
        │   ├── 2321 Periodic/Other
        │   ├── 2322 Cepheid
        │   ├── 2323 RR Lyrae
        │   ├── 2324 Delta Scuti
        │   ├── 2325 EB
        │   └── 2326 LPV/Mira
        └── 2330 Non-Periodic
            ├── 2331 Non-Periodic/Other
            └── 2332 AGN

https://github.com/LSSTDESC/elasticc/blob/main/taxonomy/taxonomy.ipynb
"""

import dataclasses
from typing import Any


class ConvAttr:
    """
    Convenience functions to be inherited by dataclasses.
    """

    # Allows to set attributes even on frozen dataclasses
    def set(self, name, value) -> None:
        object.__setattr__(self, name, value)

    # Makes dataclass subscriptable
    def __getitem__(self, item):
        return getattr(self, item)


class DataClassUnpack:
    """
    Class for safely unpack dictionaries into dataclasses.

    Parameters
    ----------
    data_class : Any
        Dataclass object to be initialized.
    arg_dict : dict
        Dictionary containing arguments to be passed to the dataclass.
    strict : bool, optional
        Flag to indicate if the method should raise an error if the keys in `arg_dict`
        do not match the fields in `data_class`. Default is False.

    Returns
    -------
    Any
        Instance of the dataclass `data_class`.
    """

    class_field_cache: dict = dict()

    @classmethod
    def instantiate(cls, data_class: Any, arg_dict: dict, strict: bool = False) -> Any:
        """
        Instantiates the dataclass `data_class` with arguments from `arg_dict`.

        Returns
        -------
        Any
            Instance of the dataclass `data_class`.
        """
        if data_class not in cls.class_field_cache:
            cls.class_field_cache[data_class]: set = {
                f.name for f in dataclasses.fields(data_class) if f.init
            }

        field_set: set = cls.class_field_cache[data_class]
        # Checks if all keys in arg_dict match the fields in the data_class
        if strict and not all([k in field_set for k in arg_dict.keys()]):
            raise ValueError(
                "DataClassUnpack in strict mode dosn't allow key missmatch: "
                f"{[k for k in arg_dict.keys() if k not in field_set]}."
            )
        # Only passes the arguments that match the fields in the data_class
        arg_dict_filtered = {k: v for k, v in arg_dict.items() if k in field_set}
        return data_class(**arg_dict_filtered)


@dataclasses.dataclass(frozen=True)
class TaxBase(ConvAttr):
    """
    Base abstraction class for entries in the taxonomy tree
    """

    id: int
    name: str
    level: int

    def get_ids(self):
        ids = [
            val.id
            for f in dataclasses.fields(self)
            if isinstance(val := (getattr(self, f.name)), TaxBase)
        ]
        """
        ids = list()
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if isinstance(val, TaxBase):
                ids.append(val.id)
        """

        return ids


# Hard-code taxonomy classes

# 2200 Non-recurring


@dataclasses.dataclass(frozen=True)
class SN(TaxBase):
    snother: TaxBase
    snia: TaxBase
    snibc: TaxBase
    snii: TaxBase
    snax: TaxBase
    sn91bg: TaxBase


@dataclasses.dataclass(frozen=True)
class Fast(TaxBase):
    fastother: TaxBase
    kn: TaxBase
    mdwarf: TaxBase
    nova: TaxBase
    ulens: TaxBase


@dataclasses.dataclass(frozen=True)
class Long(TaxBase):
    longother: TaxBase
    slsn: TaxBase
    tde: TaxBase
    ilot: TaxBase
    cart: TaxBase
    pisn: TaxBase


@dataclasses.dataclass(frozen=True)
class NRec(TaxBase):
    nrecother: TaxBase
    sn: SN
    fast: Fast


# 2300 Recurring


@dataclasses.dataclass(frozen=True)
class Periodic(TaxBase):
    periodicother: TaxBase
    ceph: TaxBase
    rrlyrae: TaxBase
    deltasc: TaxBase
    eb: TaxBase
    lpvmira: TaxBase


@dataclasses.dataclass(frozen=True)
class NPeriodic(TaxBase):
    nperiodicother: TaxBase
    agn: TaxBase


@dataclasses.dataclass(frozen=True)
class Rec(TaxBase):
    recother: TaxBase
    periodic: Periodic
    nperiodic: NPeriodic


# 2000 Variable


@dataclasses.dataclass(frozen=True)
class Var(TaxBase):
    varother: TaxBase
    nrec: NRec
    rec: Rec


# 1000 Static


@dataclasses.dataclass(frozen=True)
class Static(TaxBase):
    statother: TaxBase


# 0 Meta


@dataclasses.dataclass(frozen=True)
class Meta(TaxBase):
    metaother: TaxBase
    resodial: TaxBase
    noclass: TaxBase


# Root
@dataclasses.dataclass(frozen=True)
class Alert(ConvAttr):
    meta: Meta
    static: Static
    var: Var


# Instantiate based on

# 2200 Non-recurring

nrecother = TaxBase(2210, "nrecother", 3)

sn = SN(
    id=2220,
    name="sn",
    level=3,
    snother=TaxBase(2221, "snother", 4),
    snia=TaxBase(2222, "snia", 4),
    snibc=TaxBase(2223, "snibc", 4),
    snii=TaxBase(2224, "snii", 4),
    snax=TaxBase(2225, "snax", 4),
    sn91bg=TaxBase(2226, "sn91bg", 4),
)

fast = Fast(
    id=2230,
    name="fast",
    level=3,
    fastother=TaxBase(2231, "fastother", 4),
    kn=TaxBase(2232, "kn", 4),
    mdwarf=TaxBase(2233, "mdwarf", 4),
    nova=TaxBase(2234, "nova", 4),
    ulens=TaxBase(2235, "ulens", 4),
)

long = Long(
    id=2240,
    name="long",
    level=3,
    longother=TaxBase(2240, "longother", 4),
    slsn=TaxBase(2240, "slsn", 4),
    tde=TaxBase(2240, "tde", 4),
    ilot=TaxBase(2240, "ilot", 4),
    cart=TaxBase(2240, "cart", 4),
    pisn=TaxBase(2240, "pisn", 4),
)

nrec = NRec(
    id=2200,
    name="nrec",
    level=2,
    nrecother=nrecother,
    sn=sn,
    fast=fast,
)

# 2300 Recurring

recother = TaxBase(2310, "recother", 3)

periodic = Periodic(
    id=2320,
    name="periodic",
    level=3,
    periodicother=TaxBase(2321, "periodicother", 4),
    ceph=TaxBase(2322, "ceph", 4),
    rrlyrae=TaxBase(2323, "rrlyrae", 4),
    deltasc=TaxBase(2324, "deltasc", 4),
    eb=TaxBase(2325, "eb", 4),
    lpvmira=TaxBase(2326, "lpvmira", 4),
)

nperiodic = NPeriodic(
    id=2330,
    name="nperiodic",
    level=3,
    nperiodicother=TaxBase(2331, "nperiodicother", 4),
    agn=TaxBase(2332, "agn", 4),
)

rec = Rec(
    id=2300,
    name="rec",
    level=2,
    recother=recother,
    periodic=periodic,
    nperiodic=nperiodic,
)

# 2000 Variable
var = Var(
    id=2000,
    name="var",
    level=1,
    varother=TaxBase(2100, "varother", 2),
    rec=rec,
    nrec=nrec,
)

# 1000 Static
static = Static(
    id=1000, name="static", level=1, statother=TaxBase(1100, "statother", 2)
)

# 0 Meta
meta = Meta(
    id=0,
    name="meta",
    level=1,
    metaother=TaxBase(100, "metaother", 2),
    resodial=TaxBase(200, "resodial", 2),
    noclass=TaxBase(300, "noclass", 2),
)

# Root
root = Alert(
    meta=meta,
    static=static,
    var=var,
)
