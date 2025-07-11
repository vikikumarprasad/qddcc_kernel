import numpy as np

from sklearn.metrics import r2_score, mean_squared_error
from squlearn.kernel.qsvr import QSVR
from squlearn.kernel.qgpr import QGPR
from squlearn.kernel import QKRR

from squlearn.kernel.loss.kernel_loss_base import KernelLossBase
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

class R2_Score(KernelLossBase):
    r"""
    Negative log likelihood loss function.
    This class can be used to compute the negative log likelihood loss function
    for a given quantum kernel
    :math:`K_{\theta}` with variational parameters :math:`\theta`.
    The definition of the function is taken from Equation 5.8 Chapter 5.4 of Ref. [1].

    The log-likelihood function is defined as:

    .. math::

        L(\theta) = -\frac{1}{2} log(|K_{\theta} + \sigmaI|)-\frac{1}{2} y^{T}(K_{\theta}
        + \sigmaI)^{-1}y-\frac{n}{2} log(2\pi)

    Args:
        sigma: (float), default=0.0: Hyperparameter for the regularization strength.

    References
    ----------
        [1]: `Carl E. Rasmussen and Christopher K.I. Williams,
        "Gaussian Processes for Machine Learning",
        MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

    Methods:
    --------
    """

    def __init__(self, sigma=0.0):
        super().__init__()
        self._sigma = sigma

    def compute(
        self,
        parameter_values: np.ndarray,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:


        if self._quantum_kernel is None:
            raise ValueError(
                "Quantum kernel is not set, please set the quantum kernel with set_quantum_kernel method"
            )

        # Bind training parameters
        self._quantum_kernel.assign_parameters(parameter_values)
        simplefilter("ignore", category=ConvergenceWarning)

        # range_of_values = len(data)  # e.g., numbers from 0 to 99
        # num_unique_values = min(64 * 4, range_of_values)  # Number of unique values to select

        # # Generate the list of unique random values
        # unique_values = np.random.choice(range(range_of_values), size=num_unique_values, replace=False)

        # test_data = data[unique_values]
        # test_labels = labels[unique_values]
        # print(len(test_data))

        # print(self._quantum_kernel)
        qsvr = QSVR(self._quantum_kernel, max_iter = 15) # type: ignore
        qsvr.fit(data, labels)

        return 1 - r2_score(labels, qsvr.predict(data))
    

class MSE_Score(KernelLossBase):
    r"""
    Negative log likelihood loss function.
    This class can be used to compute the negative log likelihood loss function
    for a given quantum kernel
    :math:`K_{\theta}` with variational parameters :math:`\theta`.
    The definition of the function is taken from Equation 5.8 Chapter 5.4 of Ref. [1].

    The log-likelihood function is defined as:

    .. math::

        L(\theta) = -\frac{1}{2} log(|K_{\theta} + \sigmaI|)-\frac{1}{2} y^{T}(K_{\theta}
        + \sigmaI)^{-1}y-\frac{n}{2} log(2\pi)

    Args:
        sigma: (float), default=0.0: Hyperparameter for the regularization strength.

    References
    ----------
        [1]: `Carl E. Rasmussen and Christopher K.I. Williams,
        "Gaussian Processes for Machine Learning",
        MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

    Methods:
    --------
    """

    def __init__(self, model_type, sigma=0.0, noisy=False):
        super().__init__()
        self._sigma = sigma
        self._loss_computed_count = 0
        self._loss_history = []
        self._noisy = noisy
        self._model_type = model_type

    @property
    def train_loss_history(self):
        return self._loss_history

    def compute(
        self,
        parameter_values: np.ndarray,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:

        if self._quantum_kernel is None:
            raise ValueError(
                "Quantum kernel is not set, please set the quantum kernel with set_quantum_kernel method"
            )

        # Bind training parameters
        self._quantum_kernel.assign_parameters(parameter_values)
        simplefilter("ignore", category=ConvergenceWarning)

        # range_of_values = len(data)  # e.g., numbers from 0 to 99
        # num_unique_values = min(64 * 4, range_of_values)  # Number of unique values to select

        # # Generate the list of unique random values
        # unique_values = np.random.choice(range(range_of_values), size=num_unique_values, replace=False)

        # test_data = data[unique_values]
        # test_labels = labels[unique_values]
        # print(len(test_data))

        print(self._quantum_kernel.parameters)
        if self._model_type == "qsvr":
            model = QSVR(self._quantum_kernel, max_iter = 50) # type: ignore
        elif self._model_type == "qgpr":
            model = QGPR(self._quantum_kernel, max_iter = 50)
        elif self._model_type == "qkrr":
            model = QKRR(self._quantum_kernel, max_iter = 50)
        model.fit(data, labels)

        loss = mean_squared_error(labels, model.predict(data))
        
        if (self._loss_computed_count % (len(parameter_values) * 2 + 1) == 0) or self._noisy:
            print(f"{self._loss_computed_count}: {loss}")
            self._loss_history.append(loss)
        else:
            print(f"{self._loss_computed_count % (len(parameter_values) * 2)}: {loss}")

        self._loss_computed_count += 1

        return loss
