# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings


def _convert_data_dataframe(
    caller_name, data, target, feature_names, target_names, sparse_data=False
):
    pd = check_pandas_support("{} with as_frame=True".format(caller_name))
    if not sparse_data:
        try:
            data_df = pd.DataFrame(data, columns=feature_names, copy=False)
        except ValueError:
            data_df = pd.DataFrame(data, copy=False)
    else:
        data_df = pd.DataFrame.sparse.from_spmatrix(data, columns=feature_names)

    target_df = pd.DataFrame(target, columns=target_names)
    combined_df = pd.concat([data_df, target_df], axis=1)

    # additional try/except
    try:
        X = combined_df[feature_names]
        y = combined_df[target_names]
    except ValueError:
        X = combined_df[:, :-1]
        y = combined_df[target_names]
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    return combined_df, X, y


def check_pandas_support(caller_name):
    """Raise ImportError with detailed error message if pandas is not installed."""
    try:
        import pandas  # noqa

        return pandas
    except ImportError as e:
        raise ImportError("{} requires pandas.".format(caller_name)) from e


class Bunch(dict):
    """Container object exposing keys as attributes.

    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.

    Examples
    --------
    >>> from sklearn.utils import Bunch
    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

        # Map from deprecated key to warning message
        self.__dict__["_deprecated_key_to_warnings"] = {}

    def __getitem__(self, key):
        if key in self.__dict__.get("_deprecated_key_to_warnings", {}):
            warnings.warn(
                self._deprecated_key_to_warnings[key],
                FutureWarning,
            )
        return super().__getitem__(key)

    def _set_deprecated(self, value, *, new_key, deprecated_key, warning_message):
        """Set key in dictionary to be deprecated with its warning message."""
        self.__dict__["_deprecated_key_to_warnings"][deprecated_key] = warning_message
        self[new_key] = self[deprecated_key] = value

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass
