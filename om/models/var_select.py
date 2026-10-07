import itertools
from pathlib import Path

import numpy as np
import pandas as panda

from om.models.model import DiscreteDistributionModel

"""
Version 3
"""
from sklearn import preprocessing


class BayesianVarSelectModel(DiscreteDistributionModel):
    def __init__(self, X, y, a, b, c, pi):
        """
        :param X: (n x q) feature matrix of n data points, each with q features. The first column of X should always be 1
        :param y: q-vector response variable modeled as N(y; X . beta, sigma^2)
        :param a: hyper-param: sigma^2 ~ IG(a, b)
        :param b: hyper-param: sigma^2 ~ IG(a, b)
        :param c: hyper-param: beta|sigma^2, X ~ N(beta; 0, c.sigma^2(X'X)^{-1})
        :param pi: hyper-param: gamma ~ Be(gamma; pi)
        """
        x_first_column = X[:, 0]
        assert np.all(x_first_column == 1), "The first column does not consist entirely of ones."

        self.X = X
        self.y = y

        self.n = self.X.shape[0]  # num data points
        self.q = self.X.shape[1]  # num features including '1' (i.e. bias)

        assert self.y.shape[0] == self.n, "y.shape:" + str(y.shape)

        self.a = a
        self.b = b
        self.c = c
        self.pi = pi

        self.state_to_neg_log_map = {state.tobytes(): self._inner_calc_neg_log_unnormalized_prob(state) for state in
                                     self.generate_all_states()}
        min_inner_neg_log = min(self.state_to_neg_log_map.values())
        self.state_to_neg_log_map = {k: v - min_inner_neg_log for k, v in self.state_to_neg_log_map.items()}

    def calc_neg_log_unnormalized_prob(self, x):
        xb = x.tobytes()
        assert xb in self.state_to_neg_log_map, "This is not a valid state: {x}".format(x=str(x))
        return self.state_to_neg_log_map[xb]

    def get_dimension(self):
        return self.q

    def generate_all_states(self):
        """
        :return: all states start with 1
        """
        return [np.append(np.ones(1), np.array(state)) for state in itertools.product([0, 1], repeat=self.q - 1)]

    def generate_init_state(self):
        return np.append(np.array([1.]),
                         np.random.binomial(n=1, p=self.pi,
                                            size=self.q - 1))  # since the first feature is always chosen

    def _calc_new_b(self, Xs):
        """
        :param Xs: input  n x num_features
        :return: (Y' . (I - (c/(c+1) . X (X'X)^{-1} X') . Y + 2*b) )/2
               see formula 13 in the notes
        """
        Y = self.y

        Xt = Xs.transpose()
        Yt = Y.transpose()

        XtX = Xt.dot(Xs)
        XtX_inverse = np.linalg.pinv(XtX)

        A = ((self.c / (self.c + 1)) * Xs).dot(XtX_inverse).dot(Xt)
        I = np.eye(A.shape[0])
        new_b = ((Yt.dot(I - A).dot(Y)).item() + 2 * self.b) / 2.0

        assert new_b >= 0, "the result is not positive: " + str(new_b)
        return new_b

    def _inner_calc_neg_log_unnormalized_prob(self, gamma):
        assert gamma[0] == 1, "The first element of gamma should always be 1"  # since bias is always there.
        S = np.array(
            [k for k in range(self.q) if
             gamma[k] == 1])  # set of indices of non-zero gammas elements

        X_s = self.X[:, S]  # only columns in X with indexes in S

        B_s = self._calc_new_b(Xs=X_s)
        S_card = len(S)  # ||S||

        log_pr_gamma = -(S_card / 2.0) * np.log(self.c + 1) - (self.a + self.n / 2) * np.log(B_s) + \
                       S_card * np.log(self.pi) + (self.q - S_card) * np.log(1 - self.pi)

        return -log_pr_gamma


def fetch_mince_nutrition_data(
        mice_data_path='models/lifespan-merged.csv'):
    requested_path = Path(mice_data_path).expanduser()
    candidate_paths = [requested_path]
    if not requested_path.is_absolute():
        models_dir = Path(__file__).resolve().parent
        candidate_paths.extend([
            models_dir / requested_path,
            models_dir.parent / requested_path,
            models_dir / requested_path.name,
        ])

    resolved_path = next((path for path in candidate_paths if path.exists()), None)
    if resolved_path is None:
        searched = ", ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(f"Could not find mice data file. Tried: {searched}")

    lifespan = panda.read_csv(resolved_path)

    covariates = ['Dry weight food eaten (g/mouse/cage/d)',
                  'Cellulose intake (g/d)',
                  'P eaten (kJ/mse/cage/d)',
                  'C eaten (kJ/mse/cage/d)',
                  'F eaten (kJ/mse/cage/d)',
                  'Energy intake (kJ/mse/cage/d)']
    X_names = [s[:2] for s in covariates]

    XandY = lifespan[covariates + ['age at death (w)']]
    del lifespan
    XandY = XandY.replace('', np.nan)
    XandY = XandY.dropna()

    X = XandY[covariates]
    Y = XandY['age at death (w)']
    X = X.to_numpy()
    Y = Y.to_numpy()


    Y = np.expand_dims(Y, axis=1)  # to convert the 1D [y_0, y_1, ...] to 2D [[y_0], [y_1], ...]

    Prt = XandY['P eaten (kJ/mse/cage/d)'].to_numpy()
    Crb = XandY['C eaten (kJ/mse/cage/d)'].to_numpy()
    Fat = XandY['F eaten (kJ/mse/cage/d)'].to_numpy()

    Prt = np.expand_dims(Prt, axis=1)
    Crb = np.expand_dims(Crb, axis=1)
    Fat = np.expand_dims(Fat, axis=1)

    X = np.append(X, Prt * Crb, axis=1)
    X_names.extend(['PxC'])
    X = np.append(X, Prt * Fat, axis=1)
    X_names.extend(['PxF'])
    X = np.append(X, Crb * Fat, axis=1)
    X_names.extend(['CxF'])
    X = np.append(X, Prt * Crb * Fat, axis=1)
    X_names.extend(['PxCxF'])
    X = np.append(X, Prt / (Prt + Crb + Fat), axis=1)
    X_names.extend(['P/(P+C+F)'])
    X = np.append(X, Crb / (Prt + Crb + Fat), axis=1)
    X_names.extend(['C/(P+C+F)'])
    X = np.append(X, Fat / (Prt + Crb + Fat), axis=1)
    X_names.extend(['F/(P+C+F)'])

    assert len(X_names) == X.shape[1]


    return X, Y, X_names


def standardize_add_one_column(X, X_names):
    X = preprocessing.scale(X)

    X = np.append(np.array([[1.0]] * X.shape[0]), X, axis=1)  # add a 1 vector for intercepts

    if X_names:
        X_names = ['1'] + X_names
    return X, X_names


"""
gamma_1     is associated with      'Dry weight food eaten'
gamma_2     is associated with      'Cellulose intake',
gamma_3     is associated with      'P eaten'
gamma_4     is associated with      'C eaten'
gamma_5     is associated with      'F eaten'
gamma_6     is associated with      'Energy intake'
gamma 7     is associated with       'PxC'
gamma 8     is associated with       'PxF'
gamma 9     is associated with       'CxF'
gamma 10    is associated with       'PxCxF'
gamma 11    is associated with       Prt / (Prt + Crb + Fat)
gamma 12    is associated with       Crb / (Prt + Crb + Fat)
gamma 13    is associated with       Fat / (Prt + Crb + Fat)


"""
