import numpy
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
import warnings

from tslearn.utils import to_time_series_dataset, check_dims

from .kmeans import TimeSeriesKMeans
from .utils import TimeSeriesCentroidBasedClusteringMixin

__author__ = "tslearn contributors"


class TimeSeriesBisectingKMeans(
    TimeSeriesCentroidBasedClusteringMixin, ClusterMixin, BaseEstimator
):
    """Bisecting (divisive) K-means clustering for time-series data.

    The algorithm starts with all time series in a single cluster. At each
    step, one cluster is chosen according to `bisecting_strategy` and is split
    in two with :class:`TimeSeriesKMeans` (``n_clusters=2``), until
    `n_clusters` clusters are obtained. It is a top-down alternative to
    hierarchical clustering that does not need the full pairwise distance
    matrix and that works with the same metrics as
    :class:`TimeSeriesKMeans` (e.g. DTW or soft-DTW).

    Parameters
    ----------
    n_clusters : int (default: 8)
        Number of clusters to form.
    bisecting_strategy : {"biggest_inertia", "largest_cluster"} \
(default: "biggest_inertia")
        Which cluster is split at each step. "biggest_inertia" picks the
        cluster with the largest sum of squared distances to its center,
        "largest_cluster" picks the cluster with the most time series.
    max_iter : int (default: 50)
        Maximum number of iterations of the k-means algorithm for a single
        run of a split.
    tol : float (default: 1e-6)
        Inertia variation threshold used by each split.
    n_init : int (default: 1)
        Number of times the k-means algorithm is run with different centroid
        seeds for each split. The best output in terms of inertia is kept.
    metric : {"euclidean", "dtw", "softdtw"} (default: "euclidean")
        Metric used for both cluster assignment and barycenter computation.
    max_iter_barycenter : int (default: 100)
        Number of iterations of the barycenter computation. Only used if
        `metric="dtw"` or `metric="softdtw"`.
    metric_params : dict or None (default: None)
        Parameter values for the chosen metric.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
    verbose : int (default: 0)
        If nonzero, print information about the splits.
    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers of each split. If an integer
        is given, it fixes the seed. Defaults to the global numpy random
        number generator.

    Attributes
    ----------
    labels_ : numpy.ndarray of shape (n_ts,)
        Labels of each time series.
    cluster_centers_ : numpy.ndarray of shape (n_clusters, sz, d)
        Cluster centers. Shorter centers are padded with NaNs.
    n_iter_ : int
        Total number of k-means iterations performed over all the splits.
    inertia_ : float
        Average of the squared distances of the time series to their cluster
        center.
    splits_ : numpy.ndarray of shape (n_clusters - 1, 3)
        The split hierarchy. Each row is ``(parent, child_a, child_b)`` where
        the three values are node ids. Node 0 is the root (the whole
        dataset). Rows are in the order in which the splits were performed.
    leaf_nodes_ : numpy.ndarray of shape (n_clusters,)
        ``leaf_nodes_[k]`` is the node id of the cluster with label ``k``.

    Notes
    -----
    If `metric` is set to `"euclidean"`, the algorithm expects a dataset of
    equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> X = random_walks(n_ts=30, sz=16, d=1, random_state=0)
    >>> bkm = TimeSeriesBisectingKMeans(n_clusters=4, max_iter=5,
    ...                                 random_state=0).fit(X)
    >>> bkm.cluster_centers_.shape
    (4, 16, 1)
    >>> bkm.labels_.shape
    (30,)
    >>> bkm.splits_.shape
    (3, 3)
    >>> bkm.predict(X[:5]).shape
    (5,)

    See Also
    --------
    TimeSeriesKMeans : K-means clustering for time series.
    """

    def __init__(
        self,
        n_clusters=8,
        bisecting_strategy="biggest_inertia",
        max_iter=50,
        tol=1e-6,
        n_init=1,
        metric="euclidean",
        max_iter_barycenter=100,
        metric_params=None,
        n_jobs=None,
        verbose=0,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.bisecting_strategy = bisecting_strategy
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.metric = metric
        self.max_iter_barycenter = max_iter_barycenter
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _make_kmeans(self, n_clusters, rs):
        return TimeSeriesKMeans(
            n_clusters=n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            metric=self.metric,
            max_iter_barycenter=self.max_iter_barycenter,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=rs,
        )

    def _check_params(self):
        if self.bisecting_strategy not in ("biggest_inertia", "largest_cluster"):
            raise ValueError(
                "bisecting_strategy should be 'biggest_inertia' or "
                "'largest_cluster', got %r" % (self.bisecting_strategy,)
            )
        if (
            not isinstance(self.n_clusters, (int, numpy.integer))
            or self.n_clusters < 1
        ):
            raise ValueError(
                "n_clusters should be a positive integer, got %r"
                % (self.n_clusters,)
            )

    @staticmethod
    def _sum_sq_dists(km, X, k):
        dists = km._transform(X)
        return float(numpy.sum(dists[numpy.arange(X.shape[0]), k] ** 2))

    def fit(self, X, y=None):
        """Compute bisecting k-means clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        y
            Ignored
        """
        self._check_params()
        X = check_array(
            X,
            allow_nd=True,
            force_all_finite="allow-nan" if self.metric != "euclidean" else True,
        )
        X_ = to_time_series_dataset(X)
        n_ts = X_.shape[0]
        if self.n_clusters > n_ts:
            raise ValueError(
                "n_clusters=%d must be smaller than or equal to the number "
                "of time series (%d)" % (self.n_clusters, n_ts)
            )
        rs = check_random_state(self.random_state)

        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = numpy.inf
        self.n_iter_ = 0

        # Leaves of the tree: node id -> (indices, center, sum of squared dists)
        self._split_models = {}
        self._children = {}
        splits = []
        next_node = 1

        if self.n_clusters == 1:
            km = self._make_kmeans(1, rs).fit(X_)
            if km.cluster_centers_ is None:
                self._post_fit(X_, None, numpy.inf)
                return self
            self.n_iter_ = km.n_iter_
            self._finish(X_, [0], {0: numpy.arange(n_ts)}, {0: km.cluster_centers_[0]},
                         splits)
            return self

        leaves = {0: numpy.arange(n_ts)}
        centers = {}
        scores = {}
        splittable = {0: True}
        while len(leaves) < self.n_clusters:
            candidates = [
                node
                for node in leaves
                if len(leaves[node]) >= 2 and splittable.get(node, True)
            ]
            if not candidates:
                warnings.warn(
                    "Could not split further: only %d clusters were found "
                    "instead of n_clusters=%d." % (len(leaves), self.n_clusters),
                    ConvergenceWarning,
                )
                break
            if self.bisecting_strategy == "largest_cluster":
                node = max(candidates, key=lambda n: len(leaves[n]))
            else:
                # The root has no score yet: it is always split first
                node = max(candidates, key=lambda n: scores.get(n, numpy.inf))
            idx = leaves[node]
            km = self._make_kmeans(2, rs).fit(X_[idx])
            self.n_iter_ += km.n_iter_
            if km.cluster_centers_ is None or len(numpy.unique(km.labels_)) < 2:
                # e.g. identical series: this cluster cannot be split
                splittable[node] = False
                continue
            a, b = next_node, next_node + 1
            next_node += 2
            del leaves[node]
            scores.pop(node, None)
            self._split_models[node] = km
            self._children[node] = (a, b)
            splits.append((node, a, b))
            for child, k in ((a, 0), (b, 1)):
                sub = idx[km.labels_ == k]
                leaves[child] = sub
                centers[child] = km.cluster_centers_[k]
                scores[child] = self._sum_sq_dists(
                    km, X_[sub], numpy.full(len(sub), k)
                )
                splittable[child] = True
            if self.verbose:
                print("Split node %d into %d and %d" % (node, a, b))

        leaf_ids = sorted(leaves)
        if leaf_ids == [0]:
            km = self._make_kmeans(1, rs).fit(X_)
            self.n_iter_ += km.n_iter_
            centers[0] = km.cluster_centers_[0]
        self._finish(X_, leaf_ids, leaves, centers, splits)
        return self

    def _finish(self, X_, leaf_ids, leaves, centers, splits):
        n_ts = X_.shape[0]
        labels = numpy.empty(n_ts, dtype=int)
        for k, node in enumerate(leaf_ids):
            labels[leaves[node]] = k
        self.labels_ = labels
        self.leaf_nodes_ = numpy.array(leaf_ids, dtype=int)
        self._leaf_label = {node: k for k, node in enumerate(leaf_ids)}
        self.cluster_centers_ = to_time_series_dataset(
            [centers[node] for node in leaf_ids]
        )
        self.splits_ = numpy.array(splits, dtype=int).reshape((-1, 3))
        self.n_features_in_ = X_.shape[-1]
        self.inertia_ = float(
            numpy.sum(self._leaf_sq_dists(X_, labels)) / n_ts
        )

    def _leaf_sq_dists(self, X_, labels):
        """Squared distance of each series to the center of its leaf."""
        km = self._make_kmeans(self.cluster_centers_.shape[0], None)
        km.cluster_centers_ = self.cluster_centers_
        dists = km._transform(X_)
        return dists[numpy.arange(X_.shape[0]), labels] ** 2

    def _predict_nodes(self, X_):
        labels = numpy.empty(X_.shape[0], dtype=int)
        stack = [(0, numpy.arange(X_.shape[0]))]
        while stack:
            node, idx = stack.pop()
            if len(idx) == 0:
                continue
            if node not in self._split_models:
                labels[idx] = self._leaf_label[node]
                continue
            side = self._split_models[node].predict(X_[idx])
            a, b = self._children[node]
            stack.append((a, idx[side == 0]))
            stack.append((b, idx[side == 1]))
        return labels

    def fit_predict(self, X, y=None):
        """Fit the model and return the label of each time series.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        y
            Ignored

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y).labels_

    def predict(self, X):
        """Predict the cluster each time series in X belongs to.

        The time series go down the split hierarchy: at each split, they
        follow the closest of the two sub-clusters.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)
        X = check_array(
            X,
            allow_nd=True,
            force_all_finite="allow-nan" if self.metric != "euclidean" else True,
        )
        X = check_dims(
            X,
            X_fit_dims=self.cluster_centers_.shape,
            extend=True,
            check_n_features_only=(self.metric != "euclidean"),
        )
        return self._predict_nodes(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = self.metric != "euclidean"
        tags.allow_variable_length = self.metric != "euclidean"
        return tags
