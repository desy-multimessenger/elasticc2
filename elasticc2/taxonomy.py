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
from pprint import pformat
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


def set_dataclass_value(
    obj: Any, field_name: str, new_value: Any, force: bool = False
) -> Any:
    """
    Recursively search for a field in a dataclass object and change its value to
    a new value.

    Parameters
    ----------
    obj : Any
        The input dataclass object to modify.
    field_name : str
        The name of the field to modify.
    new_value : Any
        The new value to set the field to.
    force : bool
        Modify value despite the dataclass being set to frozen

    Returns
    -------
    Any
        A modified dataclass object with the specified field changed to the new value.
    """
    # If the input is not a dataclass, return it as is
    if not dataclasses.is_dataclass(obj):
        return obj

    # Set the value of the field to the new value
    if obj.__dataclass_params__.frozen and force:
        object.__setattr__(obj, field_name, new_value)
    else:
        setattr(obj, field_name, new_value)

    # Recursively modify the nested dataclass objects
    # Loop over fields
    for f in dataclasses.fields(obj):
        # Gets field
        inner_obj = getattr(obj, f.name)

        # If inner field is a dataclass go recursive
        if dataclasses.is_dataclass(inner_obj):
            if obj.__dataclass_params__.frozen and force:
                object.__setattr__(
                    obj,
                    f.name,
                    set_dataclass_value(inner_obj, field_name, new_value, force),
                )
            else:
                setattr(
                    obj,
                    f.name,
                    set_dataclass_value(inner_obj, field_name, new_value, force),
                )
        else:
            continue

    return obj


@dataclasses.dataclass(frozen=True)
class TaxBase(ConvAttr):
    """
    Base abstraction class for entries in the taxonomy tree
    """

    id: int
    name: str
    level: int

    def get_ids(self, exclude: list[int] | int = [], level: int = 4) -> list[int]:
        """
        Returns a list of class IDs based on specified taxonomy level. IDs provided by
        the exclude parameter are ignored in the output.
        """
        if not isinstance(exclude, list):
            exclude = [exclude]

        ids = []
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)

            if isinstance(val, TaxBase):
                if val.level == level and val.id not in exclude:
                    ids.append(val.id)
                else:
                    ids.extend(val.get_ids(exclude=exclude, level=level))

        return ids

    def ids_from_keys(self, keys: list[str] | str) -> list[int]:
        """
        Returns list of class IDs based on a set keys
        """
        if not isinstance(keys, list):
            keys = [keys]

        ids = []
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)

            if isinstance(val, TaxBase):
                if val.name in keys:
                    ids.append(val.id)
                else:
                    ids.extend(val.ids_from_keys(keys))
        return ids

    def keys_from_ids(self, ids: int | list[int]) -> list[str]:
        """
        Returns a list of class keys based on a set of IDs
        """
        if not isinstance(ids, list):
            ids = [ids]

        keys = []
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)

            if isinstance(val, TaxBase):
                if val.id in ids:
                    keys.append(val.name)
                else:
                    keys.extend(val.keys_from_ids(ids))
        return keys

    def to_dict(self) -> dict:
        """Convert dataclass to dictionary"""
        return dataclasses.asdict(self)

    def info(self) -> None:
        """Pretty-print dataclass contents"""
        print(pformat(self.to_dict(), sort_dicts=False))


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
    long: Long


# 2300 Recurring


@dataclasses.dataclass(frozen=True)
class Periodic(TaxBase):
    pother: TaxBase
    ceph: TaxBase
    rrlyrae: TaxBase
    deltasc: TaxBase
    eb: TaxBase
    lpvmira: TaxBase


@dataclasses.dataclass(frozen=True)
class NPeriodic(TaxBase):
    npother: TaxBase
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
    longother=TaxBase(2241, "longother", 4),
    slsn=TaxBase(2242, "slsn", 4),
    tde=TaxBase(2243, "tde", 4),
    ilot=TaxBase(2244, "ilot", 4),
    cart=TaxBase(2245, "cart", 4),
    pisn=TaxBase(2246, "pisn", 4),
)

nrec = NRec(
    id=2200,
    name="nrec",
    level=2,
    nrecother=nrecother,
    sn=sn,
    fast=fast,
    long=long,
)

# 2300 Recurring

recother = TaxBase(2310, "recother", 3)

periodic = Periodic(
    id=2320,
    name="periodic",
    level=3,
    pother=TaxBase(2321, "pother", 4),
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
    npother=TaxBase(2331, "npother", 4),
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
