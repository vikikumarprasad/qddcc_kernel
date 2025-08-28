import numpy as np

from sklearn.metrics import r2_score, mean_squared_error
from squlearn.kernel.qsvr import QSVR
from squlearn.kernel.qgpr import QGPR
from squlearn.kernel import QKRR

from squlearn.kernel.loss.kernel_loss_base import KernelLossBase
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

class MSE_Score(KernelLossBase):
    r"""
    Built off sQULearns defualt KernelLossBass class this class trains a model to minimize
    the mean squared error that a certain quantum ML model predicts for training data.

    Args:
        model_type = The ML model being used (IE QSVR, QKRR, or QGPR)
        noisy = Whether this method will print traininh data. (Default = False)
    """

    def __init__(self, model_type, noisy=False):
        super().__init__()
        # self._sigma = sigma
        self._loss_computed_count = 0

        #A array that stores the training scores can be accessed by outside methods
        self._loss_history = []
        self._noisy = noisy
        self._model_type = model_type

    @property
    def train_loss_history(self):
        return self._loss_history

    """This is called automatically by KernelOptimizer"""
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
